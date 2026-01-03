import os
import json
import asyncio
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

from google import genai
from google.genai import types
from utils import retry_with_backoff, RateLimiter, ProductionManifest
from config import config

client = genai.Client(api_key=config.gemini_api_key)
char_limiter = RateLimiter(rpm_limit=config.character_rpm)

class CharacterArchitect:
    def __init__(self, script_path: str):
        self.script_path = Path(script_path)
        self.output_dir = config.characters_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.objects_dir = config.objects_dir
        self.objects_dir.mkdir(parents=True, exist_ok=True)

        # Initialize Manifest
        manifest_path = self.output_dir.parent / "production_manifest.json"
        self.manifest = ProductionManifest(manifest_path)

        # Character name registry for deduplication
        self.character_registry = {}

    def normalize_character_name(self, name: str):
        """
        Normalizes character names to handle variants and duplicates.
        Returns canonical name and folder-safe name.
        """
        if not name:
            return None, None
            
        name_lower = name.lower().strip()        

        # Common titles to detect
        titles = ["captain", "professor", "doctor", "dr", "mr", "mrs", "miss", "sir", "lord", "lady"]

        # Split name into parts
        parts = name.strip().split()

        # Extract base name (last significant word)
        base_name = parts[-1].lower() if parts else ""

        # Check if this base name already has a canonical form
        for canonical, variants in self.character_registry.items():
            canonical_base = canonical.split()[-1].lower()
            if base_name == canonical_base:
                # Add this variant to the registry
                self.character_registry[canonical].add(name)
                folder_name = canonical.lower().replace(" ", "_")
                return canonical, folder_name

        # New character - determine canonical name
        # Prefer names with titles
        has_title = any(part.lower() in titles for part in parts[:-1])
        canonical = name if has_title or len(parts) > 1 else name

        # Initialize registry entry
        self.character_registry[canonical] = {name}
        folder_name = canonical.lower().replace(" ", "_")

        return canonical, folder_name

    def get_main_characters(self):
        """Extracts unique characters from our generated JSON script with deduplication."""
        with open(self.script_path, "r") as f:
            data = json.load(f)
            all_chars = []
            for page in data:
                for panel in page['panels']:
                    all_chars.extend(panel['characters'])

            # Normalize and deduplicate
            canonical_chars = []
            for char_name in all_chars:
                canonical, _ = self.normalize_character_name(char_name)
                if canonical and canonical not in canonical_chars:
                    canonical_chars.append(canonical)

            return canonical_chars

    @retry_with_backoff()
    async def identify_key_objects(self):
        """
        Analyzes the script to identify key recurring objects, vehicles, or locations 
        that require consistent visual representation (e.g., The Nautilus, A specific weapon).
        """
        print("🔍 Scanning script for key recurring objects...")
        with open(self.script_path, "r") as f:
            script_data = json.load(f)

        # Create a condensed version of the script for analysis
        script_text = ""
        for page in script_data:
            for panel in page['panels']:
                script_text += f"{panel['visual_description']} {panel.get('advice', {}).get('character_gear', '')}\n"

        prompt = """
        Analyze the following graphic novel script segments.
        Identify the top 3-5 most important RECURRING OBJECTS, VEHICLES, or LOCATIONS that need a consistent visual design.
        
        Criteria:
        1. It must appear multiple times or be central to the plot.
        2. It must be a specific object/place (e.g., "The Nautilus", "Nemo's Organ", "The Diving Suit"), not generic (e.g., "a chair").
        
        OUTPUT: JSON list of strings. Example: ["The Nautilus", "Underwater Rifle", "Grand Salon"]
        """

        response = await client.aio.models.generate_content(
            model=config.scripting_model_global_context, # Use a text model
            contents=[prompt, script_text[:50000]], # Cap text to avoid context limits
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema={
                    "type": "ARRAY",
                    "items": {"type": "STRING"}
                }
            )
        )
        
        objects = response.parsed
        print(f"   found: {objects}")
        return objects

    @retry_with_backoff()
    async def design_object(self, object_name: str, style: str):
        """
        Generates a visual profile and reference sheet for a key object.
        """
        safe_name = object_name.lower().replace(" ", "_")
        obj_folder = self.objects_dir / safe_name
        
        if obj_folder.exists() and (obj_folder / "metadata.json").exists():
            print(f"⏭️  Skipping {object_name} (Already designed)")
            with open(obj_folder / "metadata.json", "r") as f:
                return json.load(f)

        async with char_limiter: # Reuse the limiter or create a new one
            print(f"⚙️  Designing asset: {object_name}...")
            obj_folder.mkdir(parents=True, exist_ok=True)

            # Step 1: Generate Visual Specs
            spec_prompt = f"""
            Create a detailed visual technical specification for '{object_name}' in a '{style}' graphic novel.
            
            Provide:
            1. Visual Description (materials, textures, colors).
            2. Key Features (identifying shapes, mechanisms, or architectural details).
            3. Mood/Vibe (e.g., "ominous", "sleek", "decayed").

            Keep it visual.
            """

            spec_resp = await client.aio.models.generate_content(
                model=config.character_model_attributes,
                contents=spec_prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema={
                        "type": "OBJECT",
                        "properties": {
                            "visual_description": {"type": "STRING"},
                            "key_features": {"type": "ARRAY", "items": {"type": "STRING"}},
                            "mood": {"type": "STRING"}
                        },
                        "required": ["visual_description", "key_features"]
                    }
                )
            )
            specs = spec_resp.parsed

            # Step 2: Generate Reference Images (Object Sheet)
            print(f"📸 Generating reference images for {object_name}...")
            img_prompt = f"""
            Concept art sheet for {object_name}. {specs['visual_description']}. 
            Show: 1. Full view. 2. Detail close-up. 3. Alternate angle.
            Style: {style}. White background. Clean lines.
            """

            # Fallback logic (reusing same patterns as character)
            models_to_try = [
                config.image_model_primary,
                config.image_model_fallback,
                config.image_model_last_resort
            ]

            response = None
            last_error = None

            for model in models_to_try:
                try:
                    response = await client.aio.models.generate_content(
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
                    continue

            if response is None:
                raise last_error or Exception("All image models failed for object design")

            # Step 3: Save Assets
            paths = []
            for i, part in enumerate(response.parts):
                if part.inline_data:
                    img = part.as_image()
                    path = obj_folder / f"ref_{i}.png"
                    img.save(path)
                    paths.append(str(path))

            metadata = {
                "name": object_name,
                "description": specs['visual_description'],
                "key_features": specs['key_features'],
                "type": "object",
                "reference_images": paths
            }
            
            with open(obj_folder / "metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)

            print(f"✅ {object_name} designed! Assets saved in: {obj_folder}")
            return metadata

    @retry_with_backoff()
    async def design_character(self, char_name: str, style: str):
        # Normalize character name to get canonical name and folder
        canonical_name, folder_name = self.normalize_character_name(char_name)
        if not canonical_name:
            print(f"⚠️  Skipping invalid character name: {char_name}")
            return None

        if self.manifest.is_character_designed(canonical_name):
            print(f"⏭️  Skipping {canonical_name} (Already designed)")
            # Load and return existing metadata
            char_folder = self.output_dir / folder_name
            with open(char_folder / "metadata.json", "r") as f:
                return json.load(f)

        async with char_limiter:
            print(f"🎨 Designing visual profile for: {canonical_name}...")

            # Step 1: Generate detailed character attributes and description
            attr_prompt = f"""
            Analyze the character '{canonical_name}' for a '{style}' graphic novel and provide:

            1. A detailed physical description (facial features, clothing, color palette)
            2. Age range (e.g., "mid-30s", "elderly", "young adult")
            3. Occupation or role (e.g., "Sea Captain", "Professor", "Harpooner")
            4. Distinctive items they carry or wear (e.g., "brass telescope", "harpoon", "leather journal")

            Be specific and visual for use in AI image generation.
            """

            attr_resp = await client.aio.models.generate_content(
                model=config.character_model_attributes,
                contents=attr_prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema={
                        "type": "OBJECT",
                        "properties": {
                            "description": {"type": "STRING"},
                            "age_range": {"type": "STRING"},
                            "occupation": {"type": "STRING"},
                            "distinctive_items": {"type": "ARRAY", "items": {"type": "STRING"}}
                        },
                        "required": ["description", "age_range", "occupation", "distinctive_items"]
                    }
                )
            )

            char_attributes = attr_resp.parsed
            description = char_attributes['description']
            
            # Step 2: Generate the Reference Images (Character Sheet) with Three-Tier Fallback
            print(f"📸 Generating reference images for {canonical_name}...")

            img_prompt = f"Character sheet for {canonical_name}. {description}. Front view, side profile, and 3/4 view. White background, {style} style, high detail."

            # Three-tier fallback for image generation
            models_to_try = [
                config.image_model_primary,
                config.image_model_fallback,
                config.image_model_last_resort
            ]

            response = None
            last_error = None

            for model in models_to_try:
                try:
                    response = await client.aio.models.generate_content(
                        model=model,
                        contents=img_prompt,
                        config=types.GenerateContentConfig(
                            response_modalities=["IMAGE"],
                            image_config=types.ImageConfig(aspect_ratio="1:1")
                        )
                    )
                    break  # Success
                except Exception as e:
                    last_error = e
                    error_msg = str(e).lower()
                    if "429" in error_msg or "404" in error_msg or "not found" in error_msg:
                        print(f"⚠️ Model {model} unavailable/exhausted for character. Trying next...")
                        continue
                    else:
                        raise e

            if response is None:
                raise last_error or Exception("All image models failed for character design")

            # Step 3: Save the assets
            char_folder = self.output_dir / folder_name
            char_folder.mkdir(exist_ok=True)

            paths = []
            for i, part in enumerate(response.parts):
                if part.inline_data:
                    img = part.as_image()
                    path = char_folder / f"ref_{i}.png"
                    img.save(path)
                    paths.append(str(path))

            # Save the metadata for the Illustrator to use later
            metadata = {
                "name": canonical_name,
                "description": description,
                "age_range": char_attributes.get('age_range', 'unknown'),
                "occupation": char_attributes.get('occupation', 'unknown'),
                "distinctive_items": char_attributes.get('distinctive_items', []),
                "reference_images": paths
            }
            with open(char_folder / "metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)

            self.manifest.mark_character_designed(canonical_name)
            print(f"✅ {canonical_name} designed! Assets saved in: {char_folder}")
            return metadata

    async def design_all_assets(self, style: str):
        """Extracts and designs all characters AND key objects in the script."""
        
        # 1. Characters
        chars = self.get_main_characters()
        print(f"👥 Found {len(chars)} characters. Designing...")
        char_tasks = [self.design_character(char, style) for char in chars]
        
        # 2. Objects
        objects = await self.identify_key_objects()
        print(f"🛠️ Found {len(objects)} key objects. Designing...")
        obj_tasks = [self.design_object(obj, style) for obj in objects]

        # Run all concurrently
        all_tasks = char_tasks + obj_tasks
        return await asyncio.gather(*all_tasks)

if __name__ == "__main__":
    import asyncio
    # Point this to the script we generated in the previous step
    architect = CharacterArchitect("assets/output/20-thousand-leagues-under-the-sea_test_page.json")
    
    async def run():
        await architect.design_all_assets(style="Lush Watercolor")
            
    asyncio.run(run())