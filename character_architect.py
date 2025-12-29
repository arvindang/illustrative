import os
import json
from pathlib import Path
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

class CharacterArchitect:
    def __init__(self, script_path: str):
        self.script_path = Path(script_path)
        self.output_dir = Path("assets/output/characters")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def get_main_characters(self):
        """Extracts unique characters from our generated JSON script."""
        with open(self.script_path, "r") as f:
            data = json.load(f)
            all_chars = []
            for page in data:
                for panel in page['panels']:
                    all_chars.extend(panel['characters'])
            return list(set(all_chars))

    async def design_character(self, char_name: str, style: str):
        print(f"🎨 Designing visual profile for: {char_name}...")
        
        # Step 1: Generate a detailed physical description
        # We ask Gemini 1.5 Pro to 'imagine' the character based on the name/style
        desc_prompt = f"Provide a detailed physical description for the character '{char_name}' in a '{style}' graphic novel. Focus on facial features, distinctive clothing, and color palette. Output as a single paragraph."
        
        desc_resp = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=desc_prompt
        )
        description = desc_resp.text
        
        # Step 2: Generate the Reference Images (Character Sheet)
        # We use Gemini 2.5 Flash Image for fast, consistent creation
        print(f"📸 Generating reference images for {char_name}...")
        
        img_prompt = f"Character sheet for {char_name}. {description}. Front view, side profile, and 3/4 view. White background, {style} style, high detail."
        
        response = client.models.generate_content(
            model="gemini-2.5-flash-image",
            contents=img_prompt,
            config=types.GenerateContentConfig(
                response_modalities=["IMAGE"],
                image_config=types.ImageConfig(aspect_ratio="1:1")
            )
        )

        # Step 3: Save the assets
        char_folder = self.output_dir / char_name.lower().replace(" ", "_")
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
            "name": char_name,
            "description": description,
            "reference_images": paths
        }
        with open(char_folder / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"✅ {char_name} designed! Assets saved in: {char_folder}")
        return metadata

if __name__ == "__main__":
    import asyncio
    # Point this to the script we generated in the previous step
    architect = CharacterArchitect("assets/output/20-thousand-leagues-under-the-sea_test_page.json")
    
    async def run():
        chars = architect.get_main_characters()
        for char in chars:
            await architect.design_character(char, style="Lush Watercolor")
            
    asyncio.run(run())