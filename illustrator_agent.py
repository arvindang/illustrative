import os
import json
import asyncio
import io
from pathlib import Path
from PIL import Image
from dotenv import load_dotenv
load_dotenv()

from google.genai import types
from utils import (
    retry_with_backoff, RateLimiter, ProductionManifest,
    get_tpm_limiter, estimate_tokens_for_image, extract_token_usage,
    get_client
)
from config import config
from validators import PanelValidator, ConsistencyAuditor, ContinuityValidator

# Image generation rate limiter (lazy-initialized)
# Uses effective rate limits based on Vertex AI vs AI Studio
_image_limiter = None

def get_image_limiter() -> RateLimiter:
    """Get the image generation rate limiter with effective RPM based on config."""
    global _image_limiter
    if _image_limiter is None:
        effective_limits = config.get_effective_rate_limits()
        _image_limiter = RateLimiter(rpm_limit=effective_limits['image_rpm'])
    return _image_limiter

class IllustratorAgent:
    def __init__(self, script_path: str, style_prompt: str, base_output_dir: Path = None):
        self.script_path = Path(script_path)
        self.style_prompt = style_prompt
        
        if base_output_dir:
            self.base_dir = Path(base_output_dir)
            self.char_base_dir = self.base_dir / "characters"
            self.obj_base_dir = self.base_dir / "objects"
            self.output_base_dir = self.base_dir / "pages"
        else:
            self.char_base_dir = Path("assets/output/characters")
            self.obj_base_dir = config.objects_dir
            self.output_base_dir = Path("assets/output/pages")
            
        self.char_base_dir.mkdir(parents=True, exist_ok=True)
        self.obj_base_dir.mkdir(parents=True, exist_ok=True)
        self.output_base_dir.mkdir(parents=True, exist_ok=True)

        # Initialize Manifest
        manifest_path = self.output_base_dir.parent / "production_manifest.json"
        self.manifest = ProductionManifest(manifest_path)

        # In-memory cache for lazy-loaded PIL reference images
        self._character_cache = {}
        self._metadata_cache = {}
        self._object_cache = {}
        self._object_metadata_cache = {}

        # Pre-load list of known objects for scanning
        self.known_objects = []
        if self.obj_base_dir.exists():
            self.known_objects = [d.name for d in self.obj_base_dir.iterdir() if d.is_dir()]

        # Build robust character lookup map
        self.character_map = {}
        self._build_character_map()

        self.current_model = config.image_model_primary

        # Rate limiter for reference image generation (uses effective limits)
        effective_limits = config.get_effective_rate_limits()
        self.ref_limiter = RateLimiter(rpm_limit=effective_limits['character_rpm'])

        # Multi-pass reference generation settings
        self.ref_candidates = 3  # Number of candidates to generate
        self.enable_multi_pass_refs = True  # Toggle for multi-pass generation

        # Validation settings
        self.enable_panel_validation = True  # Validate each panel after generation
        self.enable_consistency_audit = True  # Audit character consistency per page
        self.max_regeneration_attempts = 2  # Max times to regenerate a failed panel
        self.era_constraints = ""  # Set by caller for era validation

        # Initialize validators
        self.panel_validator = PanelValidator()
        self.consistency_auditor = ConsistencyAuditor()

        # Load character arcs data if available (from enrichment pipeline)
        self.character_arcs = {}
        self.scene_states = {}
        self._load_character_arcs()

    def _load_character_arcs(self):
        """Load character arcs data from the enrichment pipeline output."""
        # Build path to character_arcs.json based on script path
        stem = self.script_path.stem.replace('_full_script', '').replace('_test_page', '')
        arcs_path = self.script_path.parent / f"{stem}_character_arcs.json"

        if arcs_path.exists():
            with open(arcs_path, "r") as f:
                self.character_arcs = json.load(f)

            # Build scene states lookup by page number
            if "scene_states" in self.character_arcs:
                for state in self.character_arcs["scene_states"]:
                    page_num = state.get("page_number", 0)
                    self.scene_states[page_num] = state
            print(f"   Loaded character arcs: {len(self.character_arcs.get('characters', []))} characters")
        else:
            print(f"   Character arcs not found at {arcs_path}, using defaults")

    def _get_scene_context(self, page_num: int, panel_data: dict) -> str:
        """
        Build scene-specific context from character arcs for panel generation.

        Args:
            page_num: Current page number
            panel_data: Panel data dict

        Returns:
            str: Scene context string for prompt injection
        """
        context_lines = []
        characters = panel_data.get('characters', [])
        advice = panel_data.get('advice', {})
        scene_type = advice.get('scene_type', '') if isinstance(advice, dict) else ''

        # Get scene state for this page
        scene_state = self.scene_states.get(page_num, {})
        char_states = scene_state.get("characters", {})
        interaction_rules = scene_state.get("interaction_rules", [])

        # Build character-specific context
        for char_name in characters:
            if char_name in char_states:
                state = char_states[char_name]
                emotional_state = state.get("emotional_state", "")
                gear = state.get("gear", [])
                notes = state.get("notes", "")

                if emotional_state:
                    context_lines.append(f"{char_name} EMOTIONAL STATE: {emotional_state}")
                if gear:
                    context_lines.append(f"{char_name} REQUIRED GEAR: {', '.join(gear)}")
                if notes:
                    context_lines.append(f"{char_name} NOTES: {notes}")

        # Add interaction rules
        if interaction_rules:
            context_lines.append(f"CHARACTER INTERACTIONS: {'; '.join(interaction_rules)}")

        # Add scene type context
        if scene_type:
            context_lines.append(f"SCENE TYPE: {scene_type}")

            # Get era-appropriate gear requirements from character arcs
            for char_name in characters:
                for char_data in self.character_arcs.get('characters', []):
                    if char_data.get('name') == char_name:
                        era_gear = char_data.get('era_appropriate_gear', {})

                        # Match scene type to gear category
                        gear_list = []
                        scene_lower = scene_type.lower()
                        if 'underwater' in scene_lower and 'underwater' in era_gear:
                            gear_list = era_gear['underwater']
                        elif 'formal' in scene_lower and 'formal' in era_gear:
                            gear_list = era_gear['formal']
                        elif 'aboard' in scene_lower and 'aboard_nautilus' in era_gear:
                            gear_list = era_gear['aboard_nautilus']

                        if gear_list:
                            context_lines.append(f"{char_name} SCENE-APPROPRIATE GEAR: {', '.join(gear_list)}")
                        break

        return "\n".join(context_lines) if context_lines else ""

    async def _select_best_reference(self, candidates: list, char_data: dict) -> int:
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
                print(f"   🏆 LLM Judge selected candidate {best_idx}: {reasoning[:80]}...")
                return max(0, min(best_idx, len(candidates) - 1))  # Clamp to valid range

        except Exception as e:
            print(f"   ⚠️ LLM Judge failed: {e}. Defaulting to first candidate.")

        return 0  # Default to first candidate if judging fails

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
                print(f"⚠️ Error reading metadata for {char_folder}: {e}")

    @retry_with_backoff()
    async def generate_character_reference(self, char_data: dict, style: str):
        """
        Generates reference images for a character from the asset manifest.
        When multi-pass is enabled, generates multiple candidates and uses LLM judge to select best.
        """
        name = char_data.get('name', 'Unknown')
        folder_name = name.lower().replace(" ", "_")
        char_folder = self.char_base_dir / folder_name

        # Check if already designed
        if self.manifest.is_character_designed(name):
            print(f"⏭️  Skipping {name} (Already designed)")
            return

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
            print(f"🎨 Generating {num_candidates} reference sheet candidate(s) for: {name}...")

            # Generate multiple candidates
            candidates = []
            for candidate_idx in range(num_candidates):
                # Acquire TPM capacity for each candidate
                estimated_tokens = estimate_tokens_for_image(img_prompt, num_reference_images=0)
                await get_tpm_limiter().acquire(estimated_tokens)

                response = None
                last_error = None

                for model in models_to_try:
                    try:
                        response = await get_client().aio.models.generate_content(
                            model=model,
                            contents=img_prompt,
                            config=types.GenerateContentConfig(
                                response_modalities=["IMAGE"],
                                image_config=types.ImageConfig(aspect_ratio="1:1")
                            )
                        )
                        break
                    except Exception as e:
                        last_error = e
                        error_msg = str(e).lower()
                        if "429" in error_msg or "404" in error_msg or "not found" in error_msg:
                            print(f"⚠️ Model {model} unavailable. Trying next...")
                            continue
                        else:
                            raise e

                if response is None:
                    if candidate_idx == 0:
                        raise last_error or Exception(f"All image models failed for {name}")
                    else:
                        print(f"   ⚠️ Could not generate candidate {candidate_idx + 1}, continuing with {len(candidates)} candidates")
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
                        print(f"   ✓ Generated candidate {candidate_idx + 1}/{num_candidates}")
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
                print(f"   📁 All {len(candidates)} candidates saved in: {candidates_dir}")

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
            print(f"✅ {name} designed! (Selected candidate {best_idx + 1}/{len(candidates)}) Assets saved in: {char_folder}")

    @retry_with_backoff()
    async def generate_object_reference(self, obj_data: dict, style: str):
        """
        Generates reference images for a key object from the asset manifest.
        """
        name = obj_data.get('name', 'Unknown')
        folder_name = name.lower().replace(" ", "_")
        obj_folder = self.obj_base_dir / folder_name

        # Check if already designed
        if (obj_folder / "metadata.json").exists():
            print(f"⏭️  Skipping {name} (Already designed)")
            return

        async with self.ref_limiter:
            print(f"⚙️ Generating reference sheet for: {name}...")
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

            for model in models_to_try:
                try:
                    response = await get_client().aio.models.generate_content(
                        model=model,
                        contents=img_prompt,
                        config=types.GenerateContentConfig(
                            response_modalities=["IMAGE"],
                            image_config=types.ImageConfig(aspect_ratio="1:1")
                        )
                    )
                    break
                except Exception as e:
                    last_error = e
                    error_msg = str(e).lower()
                    if "429" in error_msg or "404" in error_msg or "not found" in error_msg:
                        print(f"⚠️ Model {model} unavailable. Trying next...")
                        continue
                    else:
                        raise e

            if response is None:
                raise last_error or Exception(f"All image models failed for {name}")

            # Update TPM with actual INPUT usage only (output image tokens shouldn't count toward TPM)
            obj_input, obj_output = extract_token_usage(response)
            get_tpm_limiter().update_actual_usage(obj_estimated_tokens, obj_input)

            # Save images with PNG optimization
            paths = []
            for i, part in enumerate(response.parts):
                if part.inline_data:
                    img = part.as_image()
                    # Convert google.genai.types.Image to PIL Image
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

            print(f"✅ {name} designed! Assets saved in: {obj_folder}")

    async def generate_all_references(self, style: str):
        """
        Generates reference images for all characters and objects in the asset manifest.
        This should be called before panel generation.
        """
        # Load asset manifest
        manifest_path = self.script_path.parent / f"{self.script_path.stem.replace('_full_script', '').replace('_test_page', '')}_assets.json"

        if not manifest_path.exists():
            print(f"⚠️ Asset manifest not found at {manifest_path}. Skipping reference generation.")
            return

        with open(manifest_path, "r") as f:
            asset_manifest = json.load(f)

        characters = asset_manifest.get('characters', [])
        objects = asset_manifest.get('objects', [])

        print(f"\n🎨 Generating reference sheets for {len(characters)} characters and {len(objects)} objects...")

        # Generate character references (with error handling)
        char_tasks = [self.generate_character_reference(char, style) for char in characters]
        char_results = await asyncio.gather(*char_tasks, return_exceptions=True)

        # Log any character generation failures
        for i, result in enumerate(char_results):
            if isinstance(result, Exception):
                char_name = characters[i].get('name', f'Character {i+1}')
                print(f"   ❌ Character '{char_name}' reference failed: {result}")

        # Generate object references (with error handling)
        obj_tasks = [self.generate_object_reference(obj, style) for obj in objects]
        obj_results = await asyncio.gather(*obj_tasks, return_exceptions=True)

        # Log any object generation failures
        for i, result in enumerate(obj_results):
            if isinstance(result, Exception):
                obj_name = objects[i].get('name', f'Object {i+1}')
                print(f"   ❌ Object '{obj_name}' reference failed: {result}")

        # Rebuild character map after generating new references
        self._build_character_map()
        self.known_objects = [d.name for d in self.obj_base_dir.iterdir() if d.is_dir()]

        print("✅ All reference sheets generated!")

    def load_character_refs(self, char_name: str):
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
            # Fallback: legacy normalization attempt (mostly for debug)
            # from character_architect import CharacterArchitect
            # arch = CharacterArchitect("") 
            # _, folder_name = arch.normalize_character_name(char_name)
            # char_folder = self.char_base_dir / folder_name
            # if not char_folder.exists():
            print(f"⚠️ Warning: Character assets not found for '{char_name}'")
            return [], {}

        metadata_path = char_folder / "metadata.json"
        if not metadata_path.exists():
            print(f"⚠️ Warning: No metadata found for {char_name}")
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
            print(f"  📂 Lazy-loaded {len(ref_images)} refs for '{char_name}' (mapped to {meta.get('name')})")

        return ref_images, meta

    def load_object_refs(self, obj_name: str):
        """
        Lazy-loads object reference images on-demand.
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
            print(f"  📂 Lazy-loaded {len(ref_images)} refs for Object: {meta.get('name', obj_name)}")

        return ref_images, meta

    async def _call_generate_content(self, model_name: str, input_contents: list):
        """Helper to call the API with a specific model."""
        return await get_client().aio.models.generate_content(
            model=model_name,
            contents=input_contents,
            config=types.GenerateContentConfig(
                response_modalities=["IMAGE"],
                image_config=types.ImageConfig(aspect_ratio="4:3") 
            )
        )

    @retry_with_backoff(max_retries=3) # Image generation can be more brittle
    async def generate_panel(self, page_num: int, panel_data: dict, prev_panel_context: str = None, next_panel_context: str = None, prev_panel_image: Image.Image = None):
        panel_id = panel_data['panel_id']
        
        # Check if already complete
        if self.manifest.is_panel_complete(page_num, panel_id):
            print(f"   ⏭️  Skipping Page {page_num}, Panel {panel_id} (Already in manifest)")
            return

        async with get_image_limiter():
            print(f"🎨 Generating Page {page_num}, Panel {panel_id}...")

            # Determine compositional negative space based on bubble position
            bubble_pos = panel_data.get('bubble_position', 'top-left')
            composition_instruction = ""
            if bubble_pos == "top-left":
                composition_instruction = "COMPOSITION RULE: Use a cinematic 'rule of thirds' composition. Ensure the TOP-LEFT area is uncluttered, consisting only of simple background elements (like sky, a wall, or soft-focus scenery) to provide breathing room for a later text overlay. Do NOT place character faces or primary action in this corner."
            elif bubble_pos == "top-right":
                composition_instruction = "COMPOSITION RULE: Use a cinematic 'rule of thirds' composition. Ensure the TOP-RIGHT area is uncluttered, consisting only of simple background elements (like sky, a wall, or soft-focus scenery) to provide breathing room for a later text overlay. Do NOT place character faces or primary action in this corner."
            elif bubble_pos == "bottom-left":
                composition_instruction = "COMPOSITION RULE: Use a cinematic 'rule of thirds' composition. Ensure the BOTTOM-LEFT area is uncluttered, consisting only of simple background elements (like sky, a wall, or soft-focus scenery) to provide breathing room for a later text overlay. Do NOT place character faces or primary action in this corner."
            elif bubble_pos == "bottom-right":
                composition_instruction = "COMPOSITION RULE: Use a cinematic 'rule of thirds' composition. Ensure the BOTTOM-RIGHT area is uncluttered, consisting only of simple background elements (like sky, a wall, or soft-focus scenery) to provide breathing room for a later text overlay. Do NOT place character faces or primary action in this corner."
            
            # 1. Construct the master prompt
            # Parse structured advice if available, fall back to string for backward compatibility
            advice_data = panel_data.get('advice', {})
            if isinstance(advice_data, dict):
                advice_str = f"""
                CONTINUITY: {advice_data.get('continuity_notes', 'N/A')}
                HISTORICAL CONSTRAINTS: {advice_data.get('historical_constraints', 'N/A')}
                CHARACTER GEAR: {advice_data.get('character_gear', 'N/A')}
                """
            else:
                # Backward compatibility with old string-based advice
                advice_str = str(advice_data)

            # Build narrative flow context
            narrative_flow = ""
            if prev_panel_context:
                narrative_flow += f"\nPREVIOUS PANEL ACTION (Sequence Context): {prev_panel_context}"
            if next_panel_context:
                narrative_flow += f"\nNEXT PANEL ACTION (Sequence Context): {next_panel_context}"
            
            # Build era constraints block if set
            era_block = ""
            if self.era_constraints:
                era_block = f"""
            MANDATORY HISTORICAL/ERA CONSTRAINTS (CRITICAL - MUST BE FOLLOWED):
            {self.era_constraints}

            UNDERWATER SCENES: If characters are underwater or in water, they MUST wear appropriate period diving equipment (brass helmets, canvas suits with metal plates, air hoses). NO bare skin underwater. NO modern SCUBA gear.
            """

            # Build scene-specific context from character arcs
            scene_context = self._get_scene_context(page_num, panel_data)
            scene_context_block = ""
            if scene_context:
                scene_context_block = f"""
            CHARACTER STATE & SCENE CONTEXT (from story analysis):
            {scene_context}
            """

            # Build continuity instruction if previous panel image is available
            continuity_instruction = ""
            if prev_panel_image is not None:
                continuity_instruction = """
            VISUAL CONTINUITY (CRITICAL): A reference image of the PREVIOUS PANEL is provided below. You MUST maintain:
            - Same art style, line weight, and rendering technique
            - Consistent lighting direction and color temperature
            - Matching environment details (architecture, props, background elements)
            - Continuous color palette and mood
            - Character scale relative to environment
            """

            master_prompt = f"""
            STYLE DIRECTIVE: {self.style_prompt}
            {era_block}
            {scene_context_block}
            {continuity_instruction}
            PANEL VISUALS: {panel_data['visual_description']}

            NARRATIVE FLOW:{narrative_flow}

            SPECIFIC GUIDANCE:
            {advice_str}
            {composition_instruction}

            REQUIREMENTS: High quality comic panel art. Maintain consistency with provided character references. ALL clothing, technology, and props MUST be era-appropriate.
            CRITICAL NEGATIVE CONSTRAINT: Do NOT render any text, words, speech bubbles, captions, or EMPTY BOUNDING BOXES/FRAMES in the image. The image must be pure text-free art without any placeholders, graphical UI elements, or white boxes. Text will be added separately in post-production.
            """
            
            # 2. Gather necessary character references for this specific panel (lazy loading)
            input_contents = [master_prompt]

            # 2a. Include previous panel image for visual continuity (environment, lighting, style)
            if prev_panel_image is not None:
                input_contents.append("\n--- PREVIOUS PANEL (Maintain environment, lighting, and style continuity) ---")
                input_contents.append(prev_panel_image)
                print(f"   (Including previous panel image for continuity)")

            present_chars = panel_data.get('characters', [])

            chars_included = []
            char_descriptions = []
            
            # A. Process Characters
            for char_name in present_chars:
                # Lazy-load character refs on-demand
                ref_images, metadata = self.load_character_refs(char_name)
                if ref_images:
                    # Add reference images
                    input_contents.extend(ref_images)
                    chars_included.append(char_name)

                    # Add visual constant description if available
                    desc = metadata.get('description', "")
                    if desc:
                        char_descriptions.append(f"CHARACTER {char_name}: {desc}")
            
            # B. Process Objects (Explicit + Keyword Search)
            objects_included = []
            processed_obj_names = set()
            vis_desc_lower = panel_data['visual_description'].lower()
            
            # 1. Explicit Objects from Script (New Method)
            explicit_objects = panel_data.get('key_objects', [])
            for obj_name in explicit_objects:
                # Try to map the script name to a folder name
                # We can reuse the keyword logic or simple normalization
                norm_name = obj_name.lower().replace(" ", "_")
                
                # Check if this maps to a known object folder
                target_folder = None
                if norm_name in self.known_objects:
                    target_folder = norm_name
                else:
                    # Try partial matching against known objects
                    for ko in self.known_objects:
                        if ko in norm_name or norm_name in ko:
                            target_folder = ko
                            break
                
                if target_folder and target_folder not in processed_obj_names:
                    ref_images, metadata = self.load_object_refs(target_folder)
                    if ref_images:
                        input_contents.extend(ref_images)
                        real_name = metadata.get('name', obj_name)
                        objects_included.append(real_name)
                        processed_obj_names.add(target_folder)
                        
                        desc = metadata.get('description', "")
                        condition = metadata.get('condition', "")
                        material = metadata.get('material_context', "")
                        
                        full_desc = f"OBJECT {real_name}: {desc}"
                        if condition:
                            full_desc += f" (Condition: {condition})"
                        if material:
                            full_desc += f" (Material: {material})"
                        
                        char_descriptions.append(full_desc)

            # 2. Keyword Search Fallback (Legacy Method)
            for obj_dir_name in self.known_objects:
                if obj_dir_name in processed_obj_names:
                    continue

                # Check if the object name (or a clean version of it) is in the description
                clean_name = obj_dir_name.replace("_", " ")
                
                # Simple loose matching
                if clean_name in vis_desc_lower or obj_dir_name in vis_desc_lower:
                    ref_images, metadata = self.load_object_refs(obj_dir_name)
                    if ref_images:
                        input_contents.extend(ref_images)
                        real_name = metadata.get('name', clean_name)
                        objects_included.append(real_name)
                        processed_obj_names.add(obj_dir_name)
                        
                        desc = metadata.get('description', "")
                        # Reuse metadata loading logic if we want consistency
                        char_descriptions.append(f"OBJECT {real_name}: {desc}")

            if char_descriptions:
                desc_block = "\n".join(char_descriptions)
                master_prompt += f"\n\nVISUAL REFERENCES:\n{desc_block}"
                # Re-update the first part of input_contents which is the prompt
                input_contents[0] = master_prompt
            
            if chars_included:
                print(f"   (Including refs for Chars: {', '.join(chars_included)})")
            if objects_included:
                print(f"   (Including refs for Objects: {', '.join(objects_included)})")

            # Acquire TPM capacity for panel generation
            # Count reference images (everything after the prompt in input_contents)
            # This includes: previous panel image (if any), character refs, object refs
            num_ref_images = len(input_contents) - 1
            panel_estimated_tokens = estimate_tokens_for_image(master_prompt, num_reference_images=num_ref_images)
            if prev_panel_image is not None:
                # Previous panel image adds ~1500 tokens (counted in num_ref_images but log it)
                print(f"   (Token estimate includes {num_ref_images} reference images)")
            await get_tpm_limiter().acquire(panel_estimated_tokens)

            # 3. Call the API with Three-Tier Fallback Logic
            models_to_try = [
                config.image_model_primary,
                config.image_model_fallback,
                config.image_model_last_resort
            ]

            response = None
            last_error = None

            for model in models_to_try:
                try:
                    self.current_model = model
                    response = await self._call_generate_content(model, input_contents)
                    break  # Success, exit the loop
                except Exception as e:
                    last_error = e
                    error_msg = str(e).lower()
                    # Check for 429 Resource Exhausted or model not found
                    if "429" in error_msg or "404" in error_msg or "not found" in error_msg:
                        print(f"⚠️ Model {model} unavailable/exhausted. Trying next fallback...")
                        continue
                    else:
                        raise e  # Re-raise non-fallback errors

            if response is None:
                raise last_error or Exception("All image models failed")

            # Update TPM with actual INPUT usage only (output image tokens shouldn't count toward TPM)
            panel_input, panel_output = extract_token_usage(response)
            get_tpm_limiter().update_actual_usage(panel_estimated_tokens, panel_input)

            # 4. Save the output
            page_dir = self.output_base_dir / f"page_{page_num}"
            page_dir.mkdir(exist_ok=True)

            for i, part in enumerate(response.parts):
                if part.inline_data:
                    img = part.as_image()
                    # Convert google.genai.types.Image to PIL Image
                    pil_img = Image.open(io.BytesIO(img.image_bytes))

                    # 5. Post-generation validation (if enabled)
                    if self.enable_panel_validation:
                        print(f"   🔍 Validating panel...")
                        validation_result = await self.panel_validator.validate_panel(
                            pil_img,
                            panel_data,
                            era_context=self.era_constraints
                        )

                        if not validation_result.passed:
                            issues_str = "; ".join(validation_result.issues[:3])
                            print(f"   ⚠️ Validation issues ({validation_result.severity}): {issues_str}")

                            # Log validation issues but don't block for now
                            # Future: implement regeneration for high-severity failures
                            if validation_result.should_regenerate:
                                print(f"   📝 Panel flagged for potential regeneration (severity: {validation_result.severity})")
                        else:
                            print(f"   ✓ Validation passed")

                    output_path = page_dir / f"panel_{panel_id}.png"
                    pil_img.save(output_path, format="PNG", optimize=True)
                    print(f"   ✅ Saved: {output_path} (via {self.current_model})")

                    # Mark as complete in manifest
                    self.manifest.mark_panel_complete(page_num, panel_id)

                    # Return the image for consistency auditing
                    return pil_img 

    async def run_production(self):
        """
        Main loop to process the entire script.
        Character references are lazy-loaded on-demand to reduce memory usage.
        Includes per-page consistency auditing when enabled.
        """
        # 1. Load script
        with open(self.script_path, "r") as f:
            script_data = json.load(f)

        # 2. Flatten all panels to a single list with context
        all_panels = []
        for page in script_data:
            page_num = page['page_number']
            for panel in page['panels']:
                all_panels.append({
                    'page_num': page_num,
                    'panel_data': panel
                })

        total_panels = len(all_panels)
        total_pages = len(script_data)
        print(f"🎨 Generating {total_panels} panels (from {total_pages} pages)...")

        # 3. Process page by page for consistency auditing
        consistency_issues = []

        # Track the last successfully generated panel image for continuity
        last_panel_image = None

        for page in script_data:
            page_num = page['page_number']
            panels = page['panels']

            # Get all characters for this page
            page_characters = set()
            for panel in panels:
                page_characters.update(panel.get('characters', []))

            print(f"\n📄 Processing Page {page_num} ({len(panels)} panels)...")

            # Process panels SEQUENTIALLY for visual continuity
            # Each panel receives the previous panel's image for style/environment consistency
            page_images = []
            for panel_idx, panel in enumerate(panels):
                # Calculate global index for context
                global_idx = sum(len(p['panels']) for p in script_data if p['page_number'] < page_num) + panel_idx

                # Determine Previous Context (text description)
                prev_context = None
                if global_idx > 0:
                    prev_item = all_panels[global_idx - 1]
                    prev_desc = prev_item['panel_data'].get('visual_description', '')
                    prev_page = prev_item['page_num']
                    context_prefix = "[PREVIOUS PAGE FINAL PANEL] " if prev_page != page_num else ""
                    prev_context = f"{context_prefix}{prev_desc}"

                # Determine Next Context (text description)
                next_context = None
                if global_idx < total_panels - 1:
                    next_item = all_panels[global_idx + 1]
                    next_desc = next_item['panel_data'].get('visual_description', '')
                    next_page = next_item['page_num']
                    context_prefix = "[NEXT PAGE FIRST PANEL] " if next_page != page_num else ""
                    next_context = f"{context_prefix}{next_desc}"

                # Generate panel with previous panel image for visual continuity
                try:
                    result = await self.generate_panel(
                        page_num,
                        panel,
                        prev_context,
                        next_context,
                        prev_panel_image=last_panel_image
                    )
                    if result is not None:
                        page_images.append(result)
                        last_panel_image = result  # Update for next panel
                    else:
                        page_images.append(None)
                except Exception as e:
                    panel_id = panel.get('panel_id', panel_idx + 1)
                    print(f"      ❌ Panel {panel_id} failed: {e}")
                    page_images.append(None)
                    # Don't update last_panel_image on failure - keep previous good image

            # Filter out None values for consistency audit
            page_images = [img for img in page_images if img is not None]

            # 4. Cross-panel consistency audit (if enabled)
            if self.enable_consistency_audit and len(page_images) >= 2 and page_characters:
                print(f"   🔍 Auditing character consistency for Page {page_num}...")
                audit_result = await self.consistency_auditor.audit_page_consistency(
                    page_images,
                    list(page_characters)
                )

                if not audit_result.consistent:
                    print(f"   ⚠️ Consistency issues detected on Page {page_num}:")
                    for char, issues in audit_result.character_issues.items():
                        for issue in issues[:2]:  # Limit output
                            print(f"      - {char}: {issue}")

                    consistency_issues.append({
                        "page": page_num,
                        "issues": audit_result.character_issues,
                        "recommendations": audit_result.recommendations
                    })
                else:
                    print(f"   ✓ Character consistency OK for Page {page_num}")

        # 5. Summary
        print(f"\n✅ Production Complete! All {total_panels} panels generated.")

        if consistency_issues:
            print(f"\n⚠️ {len(consistency_issues)} page(s) have consistency issues:")
            for issue in consistency_issues:
                print(f"   - Page {issue['page']}: {len(issue['issues'])} character(s) affected")

            # Save consistency report
            report_path = self.output_base_dir.parent / "consistency_report.json"
            with open(report_path, "w") as f:
                json.dump(consistency_issues, f, indent=2)
            print(f"   📝 Detailed report saved to: {report_path}")

if __name__ == "__main__":
    # Configuration
    # Point to your generated script JSON
    SCRIPT_FILE = "assets/output/20-thousand-leagues-under-the-sea_full_script.json"
    
    # Define the overarching style. This should match what you used in the Scripting Agent.
    # Be Descriptive! This is appended to every single panel prompt.
    GLOBAL_STYLE = "Lush Watercolor comic book art. Dreamlike quality, soft color bleeds, visible paper texture. Ethereal lighting."

    agent = IllustratorAgent(SCRIPT_FILE, GLOBAL_STYLE)
    asyncio.run(agent.run_production())