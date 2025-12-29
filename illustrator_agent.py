import os
import json
import asyncio
from pathlib import Path
from PIL import Image
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

# Initialize Gemini Client
# Using Gemini 3 Pro Image for maximum visual quality and consistency adherence.
# For faster/cheaper batches, switch model to "gemini-2.5-flash-image"
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
IMAGE_MODEL = "gemini-3-pro-image-preview"

class IllustratorAgent:
    def __init__(self, script_path: str, style_prompt: str):
        self.script_path = Path(script_path)
        self.style_prompt = style_prompt
        self.char_base_dir = Path("assets/output/characters")
        # Output path for final pages
        self.output_base_dir = Path("assets/output/pages")
        self.output_base_dir.mkdir(parents=True, exist_ok=True)
        
        # In-memory bank of loaded PIL reference images
        self.character_bank = {} 

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

    async def generate_panel(self, page_num: int, panel_data: dict):
        panel_id = panel_data['panel_id']
        print(f"🎨 Generating Page {page_num}, Panel {panel_id}...")

        # Determine compositional negative space based on bubble position
        bubble_pos = panel_data.get('bubble_position', 'top-left')
        composition_instruction = ""
        if bubble_pos == "top-left":
            composition_instruction = "COMPOSITION RULE: Leave empty negative space in the TOP-LEFT corner for a speech bubble. Frame main action away from this corner."
        elif bubble_pos == "top-right":
            composition_instruction = "COMPOSITION RULE: Leave empty negative space in the TOP-RIGHT corner for a speech bubble. Frame main action away from this corner."
        elif bubble_pos == "bottom-left":
            composition_instruction = "COMPOSITION RULE: Leave empty negative space in the BOTTOM-LEFT corner for a speech bubble. Frame main action away from this corner."
        elif bubble_pos == "bottom-right":
            composition_instruction = "COMPOSITION RULE: Leave empty negative space in the BOTTOM-RIGHT corner for a speech bubble. Frame main action away from this corner."

        # 1. Construct the master prompt
        # Combine style, the specific panel description, and context.
        master_prompt = f"""
        STYLE DIRECTIVE: {self.style_prompt}
        
        PANEL VISUALS: {panel_data['visual_description']}
        {composition_instruction}
        
        CONTEXT (Dialogue occurring): "{panel_data['dialogue']}"
        
        REQUIREMENTS: High quality comic panel art. Maintain consistency with provided character references.
        CRITICAL NEGATIVE CONSTRAINT: Do NOT render any text, words, speech bubbles, or captions in the image. The image must be text-free art only.
        """

        # 2. Gather necessary character references for this specific panel
        # We only send references for characters who are actually present.
        input_contents = [master_prompt]
        present_chars = panel_data.get('characters', [])
        
        chars_included = []
        for char_name in present_chars:
            # Fuzzy matching could be added here if names aren't perfectly exact
            if char_name in self.character_bank:
                # Add all reference images for this character to the input list
                input_contents.extend(self.character_bank[char_name])
                chars_included.append(char_name)
        
        if chars_included:
            print(f"   (Including references for: {', '.join(chars_included)})")

        # 3. Call the API
        try:
            response = client.models.generate_content(
                model=IMAGE_MODEL,
                contents=input_contents,
                config=types.GenerateContentConfig(
                    response_modalities=["IMAGE"],
                    # Default comic panel aspect ratio. Change as needed (e.g., "16:9" for wide shots)
                    image_config=types.ImageConfig(aspect_ratio="4:3") 
                )
            )

            # 4. Save the output
            page_dir = self.output_base_dir / f"page_{page_num}"
            page_dir.mkdir(exist_ok=True)
            
            for i, part in enumerate(response.parts):
                if part.inline_data:
                    img = part.as_image()
                    # Naming convention: page_X_panel_Y.png
                    output_path = page_dir / f"panel_{panel_id}.png"
                    img.save(output_path)
                    print(f"   ✅ Saved: {output_path}")
                    # We only expect one image per generation call
                    break 

        except Exception as e:
            print(f"❌ Error generating panel {panel_id}: {e}")
            # In a production batch system, you would implement retry logic here.

    async def run_production(self):
        """Main loop to process the entire script."""
        # 1. Load assets
        self.load_character_bank()
        
        # 2. Load script
        with open(self.script_path, "r") as f:
            script_data = json.load(f)

        # 3. Iterate through pages and panels
        # NOTE: In a massive batch scenario, use a task queue (like Celery) here 
        # instead of a simple loop to avoid timeouts.
        for page in script_data:
            page_num = page['page_number']
            print(f"\n--- Starting Page {page_num} ---")
            # Process panels sequentially so we don't hit rate limits instantly
            for panel in page['panels']:
                await self.generate_panel(page_num, panel)
                # Optional: small sleep to be nice to the API if not using Batch API endpoint
                # await asyncio.sleep(1) 

if __name__ == "__main__":
    # Configuration
    # Point to your generated script JSON
    SCRIPT_FILE = "assets/output/20-thousand-leagues-under-the-sea_test_page.json"
    
    # Define the overarching style. This should match what you used in the Scripting Agent.
    # Be Descriptive! This is appended to every single panel prompt.
    GLOBAL_STYLE = "Lush Watercolor comic book art. Dreamlike quality, soft color bleeds, visible paper texture. Ethereal lighting."

    agent = IllustratorAgent(SCRIPT_FILE, GLOBAL_STYLE)
    asyncio.run(agent.run_production())