"""
Reference Agent: Character and object reference sheet generation.

Handles:
- Character reference sheet generation (multi-pass with LLM judge)
- Object reference sheet generation
- Reference image loading and caching
- Character name normalization and mapping

This agent is fully parallelizable per character/object.
"""

import json
import io
import asyncio
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from PIL import Image

from google.genai import types
from utils import (
    retry_with_backoff, RateLimiter, ProductionManifest,
    get_tpm_limiter, estimate_tokens_for_image, extract_token_usage,
    get_client, PrimaryModelQuotaExhaustedError
)
from config import config


class ReferenceAgent:
    """
    Generates and manages character and object reference sheets.

    This agent is designed for parallel execution - each character/object
    can be generated independently.
    """

    def __init__(
        self,
        assets_path: str = None,
        base_output_dir: Path = None,
        style_prompt: str = ""
    ):
        """
        Initialize the ReferenceAgent.

        Args:
            assets_path: Path to the asset manifest JSON file
            base_output_dir: Base directory for output (characters/, objects/)
            style_prompt: Style directive for image generation
        """
        self.assets_path = Path(assets_path) if assets_path else None
        self.style_prompt = style_prompt

        if base_output_dir:
            self.base_dir = Path(base_output_dir)
            self.char_base_dir = self.base_dir / "characters"
            self.obj_base_dir = self.base_dir / "objects"
        else:
            self.char_base_dir = Path("assets/output/characters")
            self.obj_base_dir = config.objects_dir

        self.char_base_dir.mkdir(parents=True, exist_ok=True)
        self.obj_base_dir.mkdir(parents=True, exist_ok=True)

        # Initialize Manifest
        manifest_path = self.char_base_dir.parent / "production_manifest.json"
        self.manifest = ProductionManifest(manifest_path)

        # In-memory cache for lazy-loaded PIL reference images
        self._character_cache: Dict[str, List[Image.Image]] = {}
        self._metadata_cache: Dict[str, Dict] = {}
        self._object_cache: Dict[str, List[Image.Image]] = {}
        self._object_metadata_cache: Dict[str, Dict] = {}

        # Pre-load list of known objects for scanning
        self.known_objects: List[str] = []
        if self.obj_base_dir.exists():
            self.known_objects = [d.name for d in self.obj_base_dir.iterdir() if d.is_dir()]

        # Build robust character lookup map
        self.character_map: Dict[str, Path] = {}
        self._build_character_map()

        # Rate limiter for reference image generation
        effective_limits = config.get_effective_rate_limits()
        self.ref_limiter = RateLimiter(rpm_limit=effective_limits['character_rpm'])

        # Multi-pass reference generation settings
        self.ref_candidates = 3  # Number of candidates to generate
        self.enable_multi_pass_refs = True  # Toggle for multi-pass generation

    async def run(self, style: str = None) -> None:
        """
        Generate all pending references from the asset manifest.

        This is the main entry point following the agent interface pattern.

        Args:
            style: Style directive for image generation (overrides constructor)
        """
        if style:
            self.style_prompt = style

        await self.generate_all_references(self.style_prompt)

    def _build_character_map(self):
        """
        Scans the character directory to build a lookup map.
        Maps: 'Canonical Name', 'Folder Name', and 'Last Name' -> Folder Path
        Example: "Professor Aronnax", "professor_aronnax", "Aronnax" -> /path/to/professor_aronnax
        """
        if not self.char_base_dir.exists():
            return

        for char_folder in self.char_base_dir.iterdir():
            if not char_folder.is_dir():
                continue

            metadata_path = char_folder / "metadata.json"
            if not metadata_path.exists():
                continue

            try:
                with open(metadata_path, "r") as f:
                    meta = json.load(f)

                folder_name = char_folder.name
                canonical_name = meta.get('name', '').lower()

                # 1. Map folder name (exact match)
                self.character_map[folder_name.lower()] = char_folder

                # 2. Map canonical name (exact match)
                if canonical_name:
                    self.character_map[canonical_name] = char_folder

                    # 3. Map last name/single name variants
                    parts = canonical_name.split()
                    if len(parts) > 1:
                        last_name = parts[-1]
                        # Only map last name if it's unique (simple collision avoidance)
                        if last_name not in self.character_map:
                            self.character_map[last_name] = char_folder
            except Exception as e:
                print(f"Warning: Error reading metadata for {char_folder}: {e}")

    async def _select_best_reference(self, candidates: List[Image.Image], char_data: dict) -> int:
        """
        Uses an LLM judge to select the best reference sheet from multiple candidates.

        Args:
            candidates: List of PIL Images (reference sheet candidates)
            char_data: Character data dict with name, description, distinctive_items, etc.

        Returns:
            int: Index of the best candidate (0-indexed)
        """
        if len(candidates) <= 1:
            return 0

        name = char_data.get('name', 'Unknown')
        description = char_data.get('description', '')
        distinctive_items = char_data.get('distinctive_items', [])
        age_range = char_data.get('age_range', '')
        occupation = char_data.get('occupation', '')

        items_str = ", ".join(distinctive_items) if distinctive_items else "none specified"

        judge_prompt = f"""
You are a quality assurance expert for character reference sheets in graphic novels.

TASK: Select the BEST reference sheet from the candidates shown below.

CHARACTER REQUIREMENTS:
- Name: {name}
- Description: {description}
- Age Range: {age_range}
- Occupation: {occupation}
- Distinctive Items (MUST be present): {items_str}

EVALUATION CRITERIA (in order of importance):
1. CONSISTENCY: Are all views (front, side, 3/4) showing the SAME character? Face, clothing, and items must match across all angles.
2. DISTINCTIVE ITEMS: Are ALL distinctive items clearly visible and accurate?
3. DESCRIPTION MATCH: Does the character match the physical description (facial features, clothing, age)?
4. QUALITY: Is the artwork clear, detailed, and usable as a reference?
5. STYLE CONSISTENCY: Is the art style consistent across all views?

Analyze each candidate and respond with ONLY a JSON object:
{{"best_index": <0-based index of best candidate>, "reasoning": "<brief explanation>"}}
"""

        # Build input with all candidate images
        input_contents = [judge_prompt]
        for i, img in enumerate(candidates):
            input_contents.append(f"\n--- CANDIDATE {i} ---")
            input_contents.append(img)

        try:
            response = await get_client().aio.models.generate_content(
                model=config.scripting_model_page_script,  # Use text model for judging
                contents=input_contents,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema={
                        "type": "OBJECT",
                        "properties": {
                            "best_index": {"type": "INTEGER"},
                            "reasoning": {"type": "STRING"}
                        },
                        "required": ["best_index"]
                    }
                )
            )

            result = response.parsed
            if result and 'best_index' in result:
                best_idx = result['best_index']
                reasoning = result.get('reasoning', 'No reasoning provided')
                print(f"   LLM Judge selected candidate {best_idx}: {reasoning[:80]}...")
                return max(0, min(best_idx, len(candidates) - 1))  # Clamp to valid range

        except Exception as e:
            print(f"   Warning: LLM Judge failed: {e}. Defaulting to first candidate.")

        return 0  # Default to first candidate if judging fails

    @retry_with_backoff()
    async def generate_character_reference(self, char_data: dict, style: str) -> Optional[Path]:
        """
        Generates reference images for a character from the asset manifest.
        When multi-pass is enabled, generates multiple candidates and uses LLM judge to select best.

        Args:
            char_data: Character data dict with name, description, etc.
            style: Style directive for image generation

        Returns:
            Path to the character folder, or None if skipped/failed
        """
        name = char_data.get('name', 'Unknown')
        folder_name = name.lower().replace(" ", "_")
        char_folder = self.char_base_dir / folder_name

        # Check if already designed
        if self.manifest.is_character_designed(name):
            print(f"Skipping {name} (Already designed)")
            return char_folder

        async with self.ref_limiter:
            char_folder.mkdir(parents=True, exist_ok=True)
            description = char_data.get('description', '')
            img_prompt = f"Character sheet for {name}. {description}. Front view, side profile, and 3/4 view. White background, {style} style, high detail."

            # Three-tier fallback models
            models_to_try = [
                config.image_model_primary,
                config.image_model_fallback,
                config.image_model_last_resort
            ]

            # Determine number of candidates to generate
            num_candidates = self.ref_candidates if self.enable_multi_pass_refs else 1
            print(f"Generating {num_candidates} reference sheet candidate(s) for: {name}...")

            # Generate multiple candidates
            candidates = []
            for candidate_idx in range(num_candidates):
                # Acquire TPM capacity for each candidate
                estimated_tokens = estimate_tokens_for_image(img_prompt, num_reference_images=0)
                await get_tpm_limiter().acquire(estimated_tokens)

                response = None
                last_error = None

                # Debug: Log rate limiter state
                tpm_limiter = get_tpm_limiter()
                print(f"   📊 TPM: {tpm_limiter.get_current_usage():,}/{tpm_limiter.effective_limit:,} tokens")
                print(f"   📊 Character RPM limit: {self.ref_limiter.rpm_limit}")

                for model_idx, model in enumerate(models_to_try):
                    try:
                        print(f"   🔄 Attempting model: {model}")
                        response = await get_client().aio.models.generate_content(
                            model=model,
                            contents=img_prompt,
                            config=types.GenerateContentConfig(
                                response_modalities=["IMAGE"],
                                image_config=types.ImageConfig(aspect_ratio="1:1")
                            )
                        )
                        print(f"   ✓ Model {model} succeeded")
                        break
                    except Exception as e:
                        last_error = e
                        error_msg = str(e).lower()
                        is_quota_error = "429" in error_msg or "resource_exhausted" in error_msg
                        is_not_found = "404" in error_msg or "not found" in error_msg

                        if is_quota_error or is_not_found:
                            error_type = "QUOTA EXHAUSTED (429)" if is_quota_error else "NOT FOUND (404)"
                            print(f"   ⚠️ Model {model}: {error_type}")

                            # Check if this is the primary model and config says to stop
                            if model_idx == 0 and is_quota_error and config.stop_on_primary_quota_exhausted:
                                print(f"\n🛑 STOPPING: Primary model '{model}' quota exhausted.")
                                print(f"   Config: stop_on_primary_quota_exhausted=True")
                                print(f"   Error details: {e}")
                                raise PrimaryModelQuotaExhaustedError(model, str(e))

                            print(f"   → Trying fallback model...")
                            continue
                        else:
                            print(f"   ❌ Model {model} error (non-retryable): {e}")
                            raise e

                if response is None:
                    if candidate_idx == 0:
                        raise last_error or Exception(f"All image models failed for {name}")
                    else:
                        print(f"   Could not generate candidate {candidate_idx + 1}, continuing with {len(candidates)} candidates")
                        break

                # Update TPM with actual INPUT usage
                input_tokens, output_tokens = extract_token_usage(response)
                get_tpm_limiter().update_actual_usage(estimated_tokens, input_tokens)

                # Extract PIL image from response
                for part in response.parts:
                    if part.inline_data:
                        img = part.as_image()
                        pil_img = Image.open(io.BytesIO(img.image_bytes))
                        candidates.append(pil_img)
                        print(f"   Generated candidate {candidate_idx + 1}/{num_candidates}")
                        break

            # Select best candidate using LLM judge (if multiple candidates)
            if len(candidates) > 1 and self.enable_multi_pass_refs:
                best_idx = await self._select_best_reference(candidates, char_data)
                best_image = candidates[best_idx]
            else:
                best_idx = 0
                best_image = candidates[0] if candidates else None

            if best_image is None:
                raise Exception(f"No valid reference images generated for {name}")

            # Save the best reference image
            paths = []
            path = char_folder / f"ref_0.png"
            best_image.save(path, format="PNG", optimize=True)
            paths.append(str(path))

            # Optionally save all candidates for debugging (in subdirectory)
            if len(candidates) > 1:
                candidates_dir = char_folder / "candidates"
                candidates_dir.mkdir(exist_ok=True)
                for i, candidate in enumerate(candidates):
                    candidate_path = candidates_dir / f"candidate_{i}.png"
                    candidate.save(candidate_path, format="PNG", optimize=True)
                print(f"   All {len(candidates)} candidates saved in: {candidates_dir}")

            # Save metadata
            metadata = {
                "name": name,
                "description": description,
                "age_range": char_data.get('age_range', 'unknown'),
                "occupation": char_data.get('occupation', 'unknown'),
                "distinctive_items": char_data.get('distinctive_items', []),
                "reference_images": paths,
                "generation_method": "multi_pass" if num_candidates > 1 else "single_pass",
                "candidates_evaluated": len(candidates),
                "selected_candidate": best_idx
            }
            with open(char_folder / "metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)

            self.manifest.mark_character_designed(name)
            print(f"{name} designed! (Selected candidate {best_idx + 1}/{len(candidates)}) Assets saved in: {char_folder}")

            return char_folder

    @retry_with_backoff()
    async def generate_object_reference(self, obj_data: dict, style: str) -> Optional[Path]:
        """
        Generates reference images for a key object from the asset manifest.

        Args:
            obj_data: Object data dict with name, description, etc.
            style: Style directive for image generation

        Returns:
            Path to the object folder, or None if skipped/failed
        """
        name = obj_data.get('name', 'Unknown')
        folder_name = name.lower().replace(" ", "_")
        obj_folder = self.obj_base_dir / folder_name

        # Check if already designed
        if (obj_folder / "metadata.json").exists():
            print(f"Skipping {name} (Already designed)")
            return obj_folder

        async with self.ref_limiter:
            print(f"Generating reference sheet for: {name}...")
            obj_folder.mkdir(parents=True, exist_ok=True)

            description = obj_data.get('description', '')
            key_features = obj_data.get('key_features', [])
            condition = obj_data.get('condition', '')
            material = obj_data.get('material_context', '')

            features_str = ", ".join(key_features) if key_features else ""

            img_prompt = f"Concept art sheet for {name}. {description}. {features_str}. Condition: {condition}. Materials: {material}. Show: 1. Full view. 2. Detail close-up. 3. Alternate angle. Style: {style}. White background. Clean lines."

            # Acquire TPM capacity for object reference generation
            obj_estimated_tokens = estimate_tokens_for_image(img_prompt, num_reference_images=0)
            await get_tpm_limiter().acquire(obj_estimated_tokens)

            # Three-tier fallback
            models_to_try = [
                config.image_model_primary,
                config.image_model_fallback,
                config.image_model_last_resort
            ]

            response = None
            last_error = None

            # Debug: Log rate limiter state
            tpm_limiter = get_tpm_limiter()
            print(f"   📊 TPM: {tpm_limiter.get_current_usage():,}/{tpm_limiter.effective_limit:,} tokens")
            print(f"   📊 Object RPM limit: {self.ref_limiter.rpm_limit}")

            for model_idx, model in enumerate(models_to_try):
                try:
                    print(f"   🔄 Attempting model: {model}")
                    response = await get_client().aio.models.generate_content(
                        model=model,
                        contents=img_prompt,
                        config=types.GenerateContentConfig(
                            response_modalities=["IMAGE"],
                            image_config=types.ImageConfig(aspect_ratio="1:1")
                        )
                    )
                    print(f"   ✓ Model {model} succeeded")
                    break
                except Exception as e:
                    last_error = e
                    error_msg = str(e).lower()
                    is_quota_error = "429" in error_msg or "resource_exhausted" in error_msg
                    is_not_found = "404" in error_msg or "not found" in error_msg

                    if is_quota_error or is_not_found:
                        error_type = "QUOTA EXHAUSTED (429)" if is_quota_error else "NOT FOUND (404)"
                        print(f"   ⚠️ Model {model}: {error_type}")

                        # Check if this is the primary model and config says to stop
                        if model_idx == 0 and is_quota_error and config.stop_on_primary_quota_exhausted:
                            print(f"\n🛑 STOPPING: Primary model '{model}' quota exhausted.")
                            print(f"   Config: stop_on_primary_quota_exhausted=True")
                            print(f"   Error details: {e}")
                            raise PrimaryModelQuotaExhaustedError(model, str(e))

                        print(f"   → Trying fallback model...")
                        continue
                    else:
                        print(f"   ❌ Model {model} error (non-retryable): {e}")
                        raise e

            if response is None:
                raise last_error or Exception(f"All image models failed for {name}")

            # Update TPM with actual INPUT usage only
            obj_input, obj_output = extract_token_usage(response)
            get_tpm_limiter().update_actual_usage(obj_estimated_tokens, obj_input)

            # Save images with PNG optimization
            paths = []
            for i, part in enumerate(response.parts):
                if part.inline_data:
                    img = part.as_image()
                    pil_img = Image.open(io.BytesIO(img.image_bytes))
                    path = obj_folder / f"ref_{i}.png"
                    pil_img.save(path, format="PNG", optimize=True)
                    paths.append(str(path))

            # Save metadata
            metadata = {
                "name": name,
                "description": description,
                "key_features": key_features,
                "condition": condition,
                "material_context": material,
                "type": "object",
                "reference_images": paths
            }
            with open(obj_folder / "metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)

            print(f"{name} designed! Assets saved in: {obj_folder}")
            return obj_folder

    async def generate_all_references(self, style: str) -> None:
        """
        Generates reference images for all characters and objects in the asset manifest.
        This should be called before panel generation.

        Args:
            style: Style directive for image generation
        """
        # Load asset manifest
        if not self.assets_path or not self.assets_path.exists():
            print(f"Warning: Asset manifest not found at {self.assets_path}. Skipping reference generation.")
            return

        with open(self.assets_path, "r") as f:
            asset_manifest = json.load(f)

        characters = asset_manifest.get('characters', [])
        objects = asset_manifest.get('objects', [])

        print(f"\nGenerating reference sheets for {len(characters)} characters and {len(objects)} objects...")

        # Generate character references (with error handling) - parallel
        char_tasks = [self.generate_character_reference(char, style) for char in characters]
        char_results = await asyncio.gather(*char_tasks, return_exceptions=True)

        # Log any character generation failures
        for i, result in enumerate(char_results):
            if isinstance(result, Exception):
                char_name = characters[i].get('name', f'Character {i+1}')
                print(f"   Character '{char_name}' reference failed: {result}")

        # Generate object references (with error handling) - parallel
        obj_tasks = [self.generate_object_reference(obj, style) for obj in objects]
        obj_results = await asyncio.gather(*obj_tasks, return_exceptions=True)

        # Log any object generation failures
        for i, result in enumerate(obj_results):
            if isinstance(result, Exception):
                obj_name = objects[i].get('name', f'Object {i+1}')
                print(f"   Object '{obj_name}' reference failed: {result}")

        # Rebuild character map after generating new references
        self._build_character_map()
        self.known_objects = [d.name for d in self.obj_base_dir.iterdir() if d.is_dir()]

        print("All reference sheets generated!")

    def load_character_refs(self, char_name: str) -> Tuple[List[Image.Image], Dict]:
        """
        Lazy-loads character reference images on-demand and caches them.
        Only loads what's needed for the current panel, significantly reducing memory usage.

        Args:
            char_name: The character name to load references for

        Returns:
            tuple: (list of PIL Images, metadata dict) or ([], {}) if not found
        """
        # Check cache first
        if char_name in self._character_cache:
            return self._character_cache[char_name], self._metadata_cache[char_name]

        # Resolve character folder using the map
        char_key = char_name.lower().strip()
        char_folder = self.character_map.get(char_key)

        # Try finding by partial match if exact key missing (e.g., "Dr. Aronnax" -> "aronnax")
        if not char_folder:
            parts = char_key.split()
            if parts:
                base_name = parts[-1]
                char_folder = self.character_map.get(base_name)

        if not char_folder or not char_folder.exists():
            print(f"Warning: Character assets not found for '{char_name}'")
            return [], {}

        metadata_path = char_folder / "metadata.json"
        if not metadata_path.exists():
            print(f"Warning: No metadata found for {char_name}")
            return [], {}

        # Load metadata and reference images
        with open(metadata_path, "r") as f:
            meta = json.load(f)

        ref_images = []
        for img_path_str in meta.get('reference_images', []):
            img_path = Path(img_path_str)
            if img_path.exists():
                ref_images.append(Image.open(img_path))

        # Cache for future use (cache by the REQUESTED name to speed up subsequent hits)
        self._character_cache[char_name] = ref_images
        self._metadata_cache[char_name] = meta

        if ref_images:
            print(f"  Loaded {len(ref_images)} refs for '{char_name}' (mapped to {meta.get('name')})")

        return ref_images, meta

    def load_object_refs(self, obj_name: str) -> Tuple[List[Image.Image], Dict]:
        """
        Lazy-loads object reference images on-demand.

        Args:
            obj_name: The object name (folder name) to load references for

        Returns:
            tuple: (list of PIL Images, metadata dict) or ([], {}) if not found
        """
        # Check cache
        if obj_name in self._object_cache:
            return self._object_cache[obj_name], self._object_metadata_cache[obj_name]

        obj_folder = self.obj_base_dir / obj_name
        metadata_path = obj_folder / "metadata.json"

        if not metadata_path.exists():
            return [], {}

        with open(metadata_path, "r") as f:
            meta = json.load(f)

        ref_images = []
        for img_path_str in meta.get('reference_images', []):
            img_path = Path(img_path_str)
            if img_path.exists():
                ref_images.append(Image.open(img_path))

        # Cache
        self._object_cache[obj_name] = ref_images
        self._object_metadata_cache[obj_name] = meta

        if ref_images:
            print(f"  Loaded {len(ref_images)} refs for Object: {meta.get('name', obj_name)}")

        return ref_images, meta

    def clear_cache(self):
        """Clear all cached reference images to free memory."""
        self._character_cache.clear()
        self._metadata_cache.clear()
        self._object_cache.clear()
        self._object_metadata_cache.clear()


if __name__ == "__main__":
    # Example usage
    import asyncio

    async def main():
        agent = ReferenceAgent(
            assets_path="assets/output/20-thousand-leagues-under-the-sea_assets.json",
            style_prompt="Lush Watercolor comic book art"
        )
        await agent.run()

    asyncio.run(main())
