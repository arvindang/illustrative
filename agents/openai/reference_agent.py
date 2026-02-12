"""
OpenAI Reference Agent: Character and object reference sheet generation.

Mirrors the ReferenceAgent interface from agents/reference_agent.py but uses
OpenAI's image generation API (batch or sync) instead of Gemini.

Supports:
- Batch mode: All references submitted as a single batch job (50% cost savings)
- Sync mode: Sequential generation with RPM rate limiting (for test/UI)
- Multi-pass with LLM judge selection (sync mode only)
- Resume via ProductionManifest
"""
import asyncio
import base64
import io
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image

from config import config
from utils import ProductionManifest, RateLimiter

from agents.openai.client import get_openai_client
from agents.openai.batch_manager import BatchRequest, OpenAIBatchManager


class OpenAIReferenceAgent:
    """
    Generates character and object reference sheets using OpenAI's image API.

    Drop-in replacement for ReferenceAgent when IMAGE_BACKEND=openai.
    Same directory structure and metadata format for compositor compatibility.
    """

    def __init__(
        self,
        assets_path: str = None,
        base_output_dir: Path = None,
        style_prompt: str = "",
    ):
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

        # Pre-load list of known objects
        self.known_objects: List[str] = []
        if self.obj_base_dir.exists():
            self.known_objects = [d.name for d in self.obj_base_dir.iterdir() if d.is_dir()]

        # Build character lookup map
        self.character_map: Dict[str, Path] = {}
        self._build_character_map()

        # Rate limiter for sync mode
        self.ref_limiter = RateLimiter(rpm_limit=config.openai_sync_rpm)

        # Batch manager
        self.batch_manager = OpenAIBatchManager(
            poll_interval=config.openai_batch_poll_interval,
            timeout=config.openai_batch_timeout,
        )

    def _build_character_map(self):
        """Build character name -> folder path lookup map."""
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
                canonical_name = meta.get("name", "").lower()
                self.character_map[folder_name.lower()] = char_folder
                if canonical_name:
                    self.character_map[canonical_name] = char_folder
                    parts = canonical_name.split()
                    if len(parts) > 1:
                        last_name = parts[-1]
                        if last_name not in self.character_map:
                            self.character_map[last_name] = char_folder

                # Map aliases from asset manifest
                aliases = meta.get("aliases", [])
                for alias in aliases:
                    alias_key = alias.lower().strip()
                    if alias_key and alias_key not in self.character_map:
                        self.character_map[alias_key] = char_folder
            except Exception as e:
                print(f"Warning: Error reading metadata for {char_folder}: {e}")

    def _build_ref_prompt(self, name: str, description: str, style: str) -> str:
        """Build the image generation prompt for a character reference sheet."""
        return (
            f"Character reference sheet for '{name}'. {description}. "
            f"Show: front view, side profile, and 3/4 view on a clean white background. "
            f"Style: {style}. High detail, consistent design across all angles."
        )

    def _build_obj_prompt(self, obj_data: dict, style: str) -> str:
        """Build the image generation prompt for an object reference sheet."""
        name = obj_data.get("name", "Unknown")
        description = obj_data.get("description", "")
        key_features = obj_data.get("key_features", [])
        condition = obj_data.get("condition", "")
        material = obj_data.get("material_context", "")
        features_str = ", ".join(key_features) if key_features else ""

        return (
            f"Concept art sheet for '{name}'. {description}. {features_str}. "
            f"Condition: {condition}. Materials: {material}. "
            f"Show: 1. Full view. 2. Detail close-up. 3. Alternate angle. "
            f"Style: {style}. White background, clean lines."
        )

    # ------------------------------------------------------------------ #
    #  Main entry point
    # ------------------------------------------------------------------ #

    async def generate_all_references(self, style: str) -> None:
        """
        Generate reference images for all characters and objects.

        Dispatches to batch or sync mode based on config.openai_batch_enabled.
        """
        if style:
            self.style_prompt = style

        if not self.assets_path or not self.assets_path.exists():
            print(f"Warning: Asset manifest not found at {self.assets_path}. Skipping.")
            return

        with open(self.assets_path, "r") as f:
            asset_manifest = json.load(f)

        characters = asset_manifest.get("characters", [])
        objects = asset_manifest.get("objects", [])

        print(f"\n[OpenAI] Generating reference sheets for {len(characters)} characters and {len(objects)} objects...")

        if config.openai_batch_enabled:
            await self._generate_all_batch(characters, objects)
        else:
            await self._generate_all_sync(characters, objects)

        # Rebuild maps
        self._build_character_map()
        self.known_objects = [d.name for d in self.obj_base_dir.iterdir() if d.is_dir()]
        print("[OpenAI] All reference sheets generated!")

    # ------------------------------------------------------------------ #
    #  Batch mode
    # ------------------------------------------------------------------ #

    async def _generate_all_batch(self, characters: list, objects: list) -> None:
        """Submit all pending references as a single batch job."""
        requests: List[BatchRequest] = []

        # Build character requests
        for char_data in characters:
            name = char_data.get("name", "Unknown")
            if self.manifest.is_character_designed(name):
                print(f"   Skipping {name} (already designed)")
                continue

            folder_name = name.lower().replace(" ", "_")
            prompt = self._build_ref_prompt(
                name,
                char_data.get("description", ""),
                self.style_prompt,
            )

            requests.append(BatchRequest(
                custom_id=f"char__{folder_name}",
                url="/v1/images/generations",
                body={
                    "model": config.openai_image_model_primary,
                    "prompt": prompt,
                    "size": config.openai_ref_size,
                    "quality": config.openai_ref_image_quality,
                    "n": 1,
                },
            ))

        # Build object requests
        for obj_data in objects:
            name = obj_data.get("name", "Unknown")
            folder_name = name.lower().replace(" ", "_")
            obj_folder = self.obj_base_dir / folder_name
            if (obj_folder / "metadata.json").exists():
                print(f"   Skipping {name} (already designed)")
                continue

            prompt = self._build_obj_prompt(obj_data, self.style_prompt)

            requests.append(BatchRequest(
                custom_id=f"obj__{folder_name}",
                url="/v1/images/generations",
                body={
                    "model": config.openai_image_model_primary,
                    "prompt": prompt,
                    "size": config.openai_ref_size,
                    "quality": config.openai_ref_image_quality,
                    "n": 1,
                },
            ))

        if not requests:
            print("   All references already generated, nothing to batch.")
            return

        # Run batch
        results = await self.batch_manager.run_batch(requests)

        # Process results — characters
        for char_data in characters:
            name = char_data.get("name", "Unknown")
            folder_name = name.lower().replace(" ", "_")
            custom_id = f"char__{folder_name}"

            if custom_id not in results:
                continue

            result = results[custom_id]
            if result.success:
                self._save_character_result(char_data, result.b64_image)
            else:
                print(f"   Character '{name}' batch failed: {result.error}")

        # Process results — objects
        for obj_data in objects:
            name = obj_data.get("name", "Unknown")
            folder_name = name.lower().replace(" ", "_")
            custom_id = f"obj__{folder_name}"

            if custom_id not in results:
                continue

            result = results[custom_id]
            if result.success:
                self._save_object_result(obj_data, result.b64_image)
            else:
                print(f"   Object '{name}' batch failed: {result.error}")

    # ------------------------------------------------------------------ #
    #  Sync mode
    # ------------------------------------------------------------------ #

    async def _generate_all_sync(self, characters: list, objects: list) -> None:
        """Generate references sequentially using sync API calls."""
        # Characters
        for char_data in characters:
            try:
                await self._generate_character_sync(char_data)
            except Exception as e:
                name = char_data.get("name", "?")
                print(f"   Character '{name}' failed: {e}")

        # Objects
        for obj_data in objects:
            try:
                await self._generate_object_sync(obj_data)
            except Exception as e:
                name = obj_data.get("name", "?")
                print(f"   Object '{name}' failed: {e}")

    async def _generate_character_sync(self, char_data: dict) -> Optional[Path]:
        """Generate a single character reference sheet via sync API."""
        name = char_data.get("name", "Unknown")
        if self.manifest.is_character_designed(name):
            print(f"   Skipping {name} (already designed)")
            return None

        async with self.ref_limiter:
            print(f"   [OpenAI sync] Generating reference for: {name}...")
            prompt = self._build_ref_prompt(
                name,
                char_data.get("description", ""),
                self.style_prompt,
            )

            client = get_openai_client()
            models_to_try = [
                config.openai_image_model_primary,
                config.openai_image_model_fallback,
                config.openai_image_model_last_resort,
            ]

            response = None
            for model in models_to_try:
                try:
                    response = await client.images.generate(
                        model=model,
                        prompt=prompt,
                        size=config.openai_ref_size,
                        quality=config.openai_ref_image_quality,
                        n=1,
                    )
                    break
                except Exception as e:
                    print(f"   Model {model} failed: {e}")
                    continue

            if response is None or not response.data:
                print(f"   All models failed for character '{name}'")
                return None

            b64_data = response.data[0].b64_json
            return self._save_character_result(char_data, b64_data)

    async def _generate_object_sync(self, obj_data: dict) -> Optional[Path]:
        """Generate a single object reference sheet via sync API."""
        name = obj_data.get("name", "Unknown")
        folder_name = name.lower().replace(" ", "_")
        obj_folder = self.obj_base_dir / folder_name
        if (obj_folder / "metadata.json").exists():
            print(f"   Skipping {name} (already designed)")
            return None

        async with self.ref_limiter:
            print(f"   [OpenAI sync] Generating reference for object: {name}...")
            prompt = self._build_obj_prompt(obj_data, self.style_prompt)

            client = get_openai_client()
            models_to_try = [
                config.openai_image_model_primary,
                config.openai_image_model_fallback,
                config.openai_image_model_last_resort,
            ]

            response = None
            for model in models_to_try:
                try:
                    response = await client.images.generate(
                        model=model,
                        prompt=prompt,
                        size=config.openai_ref_size,
                        quality=config.openai_ref_image_quality,
                        n=1,
                    )
                    break
                except Exception as e:
                    print(f"   Model {model} failed: {e}")
                    continue

            if response is None or not response.data:
                print(f"   All models failed for object '{name}'")
                return None

            b64_data = response.data[0].b64_json
            return self._save_object_result(obj_data, b64_data)

    # ------------------------------------------------------------------ #
    #  Saving helpers
    # ------------------------------------------------------------------ #

    def _save_character_result(self, char_data: dict, b64_data: str) -> Path:
        """Decode base64 image and save character reference to disk."""
        name = char_data.get("name", "Unknown")
        folder_name = name.lower().replace(" ", "_")
        char_folder = self.char_base_dir / folder_name
        char_folder.mkdir(parents=True, exist_ok=True)

        # Save image
        img_bytes = base64.b64decode(b64_data)
        pil_img = Image.open(io.BytesIO(img_bytes))
        ref_path = char_folder / "ref_0.png"
        pil_img.save(ref_path, format="PNG", optimize=True)

        # Save metadata (same format as Gemini ReferenceAgent)
        metadata = {
            "name": name,
            "aliases": char_data.get("aliases", []),
            "description": char_data.get("description", ""),
            "age_range": char_data.get("age_range", "unknown"),
            "occupation": char_data.get("occupation", "unknown"),
            "distinctive_items": char_data.get("distinctive_items", []),
            "reference_images": [str(ref_path)],
            "generation_method": "openai_batch" if config.openai_batch_enabled else "openai_sync",
            "candidates_evaluated": 1,
            "selected_candidate": 0,
        }
        with open(char_folder / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        self.manifest.mark_character_designed(name)
        print(f"   {name} designed! Saved to {char_folder}")
        return char_folder

    def _save_object_result(self, obj_data: dict, b64_data: str) -> Path:
        """Decode base64 image and save object reference to disk."""
        name = obj_data.get("name", "Unknown")
        folder_name = name.lower().replace(" ", "_")
        obj_folder = self.obj_base_dir / folder_name
        obj_folder.mkdir(parents=True, exist_ok=True)

        img_bytes = base64.b64decode(b64_data)
        pil_img = Image.open(io.BytesIO(img_bytes))
        ref_path = obj_folder / "ref_0.png"
        pil_img.save(ref_path, format="PNG", optimize=True)

        metadata = {
            "name": name,
            "description": obj_data.get("description", ""),
            "key_features": obj_data.get("key_features", []),
            "condition": obj_data.get("condition", ""),
            "material_context": obj_data.get("material_context", ""),
            "type": "object",
            "reference_images": [str(ref_path)],
        }
        with open(obj_folder / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"   Object '{name}' designed! Saved to {obj_folder}")
        return obj_folder

    # ------------------------------------------------------------------ #
    #  Reference loading (same interface as Gemini ReferenceAgent)
    # ------------------------------------------------------------------ #

    def load_character_refs(self, char_name: str) -> Tuple[List[Image.Image], Dict]:
        """Lazy-load character reference images, cached in memory."""
        if char_name in self._character_cache:
            return self._character_cache[char_name], self._metadata_cache[char_name]

        char_key = char_name.lower().strip()
        char_folder = self.character_map.get(char_key)

        if not char_folder:
            parts = char_key.split()
            if parts:
                char_folder = self.character_map.get(parts[-1])

        if not char_folder or not char_folder.exists():
            return [], {}

        metadata_path = char_folder / "metadata.json"
        if not metadata_path.exists():
            return [], {}

        with open(metadata_path, "r") as f:
            meta = json.load(f)

        ref_images = []
        for img_path_str in meta.get("reference_images", []):
            img_path = Path(img_path_str)
            if img_path.exists():
                ref_images.append(Image.open(img_path))

        self._character_cache[char_name] = ref_images
        self._metadata_cache[char_name] = meta
        return ref_images, meta

    def load_object_refs(self, obj_name: str) -> Tuple[List[Image.Image], Dict]:
        """Lazy-load object reference images, cached in memory."""
        if obj_name in self._object_cache:
            return self._object_cache[obj_name], self._object_metadata_cache[obj_name]

        obj_folder = self.obj_base_dir / obj_name
        metadata_path = obj_folder / "metadata.json"
        if not metadata_path.exists():
            return [], {}

        with open(metadata_path, "r") as f:
            meta = json.load(f)

        ref_images = []
        for img_path_str in meta.get("reference_images", []):
            img_path = Path(img_path_str)
            if img_path.exists():
                ref_images.append(Image.open(img_path))

        self._object_cache[obj_name] = ref_images
        self._object_metadata_cache[obj_name] = meta
        return ref_images, meta

    def clear_cache(self):
        """Clear all cached reference images to free memory."""
        self._character_cache.clear()
        self._metadata_cache.clear()
        self._object_cache.clear()
        self._object_metadata_cache.clear()
