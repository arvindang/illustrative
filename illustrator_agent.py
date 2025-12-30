import os
import json
import asyncio
import io
from pathlib import Path
from PIL import Image
from dotenv import load_dotenv
from google import genai
from google.genai import types
from utils import retry_with_backoff, RateLimiter, ProductionManifest

load_dotenv()

# Initialize Gemini Client
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
IMAGE_MODEL = "gemini-3-pro-image-preview"

# Image generation often has tighter RPM limits (e.g. 5-10 RPM)
image_limiter = RateLimiter(rpm_limit=5)

class IllustratorAgent:
    def __init__(self, script_path: str, style_prompt: str):
        self.script_path = Path(script_path)
        self.style_prompt = style_prompt
        self.char_base_dir = Path("assets/output/characters")
        self.output_base_dir = Path("assets/output/pages")
        self.output_base_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize Manifest
        manifest_path = self.output_base_dir.parent / "production_manifest.json"
        self.manifest = ProductionManifest(manifest_path)
        
        # In-memory bank of loaded PIL reference images
        self.character_bank = {} 
        self.character_bank_metadata = {}

    def load_character_bank(self):
        """
        Pre-loads all generated character reference images into memory as PIL objects.
        This is crucial for passing them to the Gemini API for consistency checking.
        """
        print("📂 Loading character reference assets into memory...")
        if not self.char_base_dir.exists():
            print("⚠️ Warning: No character assets found. Run character_architect.py first.")
            return

        for char_folder in self.char_base_dir.iterdir():
            if char_folder.is_dir():
                metadata_path = char_folder / "metadata.json"
                if metadata_path.exists():
                    with open(metadata_path, "r") as f:
                        meta = json.load(f)
                        char_name = meta['name']
                        self.character_bank_metadata[char_name] = meta
                        ref_images = []
                        # Load each reference image path listed in metadata
                        for img_path_str in meta['reference_images']:
                            img_path = Path(img_path_str)
                            if img_path.exists():
                                # Open as PIL Image
                                ref_images.append(Image.open(img_path))
                        
                        if ref_images:
                            self.character_bank[char_name] = ref_images
                            print(f"  ✅ Loaded {len(ref_images)} references for: {char_name}")
        print("--- Character Bank Ready ---")

    @retry_with_backoff(max_retries=3) # Image generation can be more brittle
    async def generate_panel(self, page_num: int, panel_data: dict):
        panel_id = panel_data['panel_id']
        
        # Check if already complete
        if self.manifest.is_panel_complete(page_num, panel_id):
            print(f"   ⏭️  Skipping Page {page_num}, Panel {panel_id} (Already in manifest)")
            return

        async with image_limiter:
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
            master_prompt = f"""
            STYLE DIRECTIVE: {self.style_prompt}
            
            PANEL VISUALS: {panel_data['visual_description']}
            SPECIFIC ADVICE: {panel_data.get('advice', '')}
            {composition_instruction}
            
            CONTEXT (Dialogue occurring): "{panel_data['dialogue']}"
            
            REQUIREMENTS: High quality comic panel art. Maintain consistency with provided character references.
            CRITICAL NEGATIVE CONSTRAINT: Do NOT render any text, words, speech bubbles, captions, or EMPTY BOUNDING BOXES/FRAMES in the image. The image must be pure text-free art without any placeholders, graphical UI elements, or white boxes.
            """
            
            # 2. Gather necessary character references for this specific panel
            input_contents = [master_prompt]
            present_chars = panel_data.get('characters', [])
            
            chars_included = []
            char_descriptions = []
            for char_name in present_chars:
                if char_name in self.character_bank:
                    # Add reference images
                    input_contents.extend(self.character_bank[char_name])
                    chars_included.append(char_name)
                    
                    # Add visual constant description if available
                    # We need to find the metadata. I'll modify load_character_bank to store it.
                    desc = self.character_bank_metadata.get(char_name, {}).get('visual_style_string', "")
                    if desc:
                        char_descriptions.append(f"CHARACTER {char_name}: {desc}")
            
            if char_descriptions:
                desc_block = "\n".join(char_descriptions)
                master_prompt += f"\n\nCHARACTER DESCRIPTIONS:\n{desc_block}"
                # Re-update the first part of input_contents which is the prompt
                input_contents[0] = master_prompt
            
            if chars_included:
                print(f"   (Including references for: {', '.join(chars_included)})")

            # 3. Call the API
            response = await client.aio.models.generate_content(
                model=IMAGE_MODEL,
                contents=input_contents,
                config=types.GenerateContentConfig(
                    response_modalities=["IMAGE"],
                    image_config=types.ImageConfig(aspect_ratio="4:3") 
                )
            )

            # 4. Save the output
            page_dir = self.output_base_dir / f"page_{page_num}"
            page_dir.mkdir(exist_ok=True)
            
            for i, part in enumerate(response.parts):
                if part.inline_data:
                    img = part.as_image()
                    output_path = page_dir / f"panel_{panel_id}.png"
                    img.save(output_path)
                    print(f"   ✅ Saved: {output_path}")

                    # Mark as complete in manifest
                    self.manifest.mark_panel_complete(page_num, panel_id)
                    
                    break 

    async def run_production(self):
        """Main loop to process the entire script."""
        # 1. Load assets
        self.load_character_bank()
        
        # 2. Load script
        with open(self.script_path, "r") as f:
            script_data = json.load(f)

        # 3. Iterate through pages and panels
        print(f"🎨 Generating panels for {len(script_data)} pages in parallel...")
        tasks = []
        for page in script_data:
            page_num = page['page_number']
            for panel in page['panels']:
                tasks.append(self.generate_panel(page_num, panel))
        
        await asyncio.gather(*tasks)

        # 4. Save the UPDATED script with optimized bubble positions
        # This ensures the CompositorAgent uses the vision-verified coordinates.
        optimized_script_path = self.script_path.parent / (self.script_path.stem + "_optimized.json")
        with open(optimized_script_path, "w") as f:
            json.dump(script_data, f, indent=2)
        print(f"\n✅ Production Complete! Optimized script saved to: {optimized_script_path}")

if __name__ == "__main__":
    # Configuration
    # Point to your generated script JSON
    SCRIPT_FILE = "assets/output/20-thousand-leagues-under-the-sea_test_page.json"
    
    # Define the overarching style. This should match what you used in the Scripting Agent.
    # Be Descriptive! This is appended to every single panel prompt.
    GLOBAL_STYLE = "Lush Watercolor comic book art. Dreamlike quality, soft color bleeds, visible paper texture. Ethereal lighting."

    agent = IllustratorAgent(SCRIPT_FILE, GLOBAL_STYLE)
    asyncio.run(agent.run_production())