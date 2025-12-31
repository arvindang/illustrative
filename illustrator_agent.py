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
from config import config

load_dotenv()

# Initialize Gemini Client
client = genai.Client(api_key=config.gemini_api_key)

# Image generation often has tighter RPM limits (e.g. 5-10 RPM)
image_limiter = RateLimiter(rpm_limit=config.image_rpm)

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
        
        # In-memory cache for lazy-loaded PIL reference images
        self._character_cache = {}
        self._metadata_cache = {}
        self.current_model = config.image_model_primary

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

        # Load from disk
        if not self.char_base_dir.exists():
            print(f"⚠️ Warning: Character folder not found for {char_name}")
            return [], {}

        # Normalize character name to folder format
        from character_architect import CharacterArchitect
        arch = CharacterArchitect("")  # Temporary instance for normalization
        _, folder_name = arch.normalize_character_name(char_name)
        char_folder = self.char_base_dir / folder_name

        if not char_folder.exists():
            print(f"⚠️ Warning: No assets found for {char_name}")
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

        # Cache for future use
        self._character_cache[char_name] = ref_images
        self._metadata_cache[char_name] = meta

        if ref_images:
            print(f"  📂 Lazy-loaded {len(ref_images)} refs for {char_name}")

        return ref_images, meta

    async def _call_generate_content(self, model_name: str, input_contents: list):
        """Helper to call the API with a specific model."""
        return await client.aio.models.generate_content(
            model=model_name,
            contents=input_contents,
            config=types.GenerateContentConfig(
                response_modalities=["IMAGE"],
                image_config=types.ImageConfig(aspect_ratio="4:3") 
            )
        )

    @retry_with_backoff(max_retries=3) # Image generation can be more brittle
    async def generate_panel(self, page_num: int, panel_data: dict, prev_panel_context: str = None, next_panel_context: str = None):
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
            
            master_prompt = f"""
            STYLE DIRECTIVE: {self.style_prompt}

            PANEL VISUALS: {panel_data['visual_description']}

            NARRATIVE FLOW:{narrative_flow}

            SPECIFIC GUIDANCE:
            {advice_str}
            {composition_instruction}

            REQUIREMENTS: High quality comic panel art. Maintain consistency with provided character references.
            CRITICAL NEGATIVE CONSTRAINT: Do NOT render any text, words, speech bubbles, captions, or EMPTY BOUNDING BOXES/FRAMES in the image. The image must be pure text-free art without any placeholders, graphical UI elements, or white boxes. Text will be added separately in post-production.
            """
            
            # 2. Gather necessary character references for this specific panel (lazy loading)
            input_contents = [master_prompt]
            present_chars = panel_data.get('characters', [])

            chars_included = []
            char_descriptions = []
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
            
            if char_descriptions:
                desc_block = "\n".join(char_descriptions)
                master_prompt += f"\n\nCHARACTER DESCRIPTIONS:\n{desc_block}"
                # Re-update the first part of input_contents which is the prompt
                input_contents[0] = master_prompt
            
            if chars_included:
                print(f"   (Including references for: {', '.join(chars_included)})")

            # 3. Call the API with Fallback Logic
            try:
                response = await self._call_generate_content(self.current_model, input_contents)
            except Exception as e:
                # Check for 429 Resource Exhausted (specifically daily limit)
                error_msg = str(e).lower()
                if "429" in error_msg and self.current_model != config.image_model_fallback:
                    print(f"⚠️ Primary model {self.current_model} exhausted quota. Falling back to {config.image_model_fallback}...")
                    self.current_model = config.image_model_fallback
                    response = await self._call_generate_content(self.current_model, input_contents)
                else:
                    raise e

            # 4. Save the output
            page_dir = self.output_base_dir / f"page_{page_num}"
            page_dir.mkdir(exist_ok=True)
            
            for i, part in enumerate(response.parts):
                if part.inline_data:
                    img = part.as_image()
                    output_path = page_dir / f"panel_{panel_id}.png"
                    img.save(output_path)
                    print(f"   ✅ Saved: {output_path} (via {self.current_model})")

                    # Mark as complete in manifest
                    self.manifest.mark_panel_complete(page_num, panel_id)
                    
                    break 

    async def run_production(self):
        """
        Main loop to process the entire script.
        Character references are lazy-loaded on-demand to reduce memory usage.
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
        
        # 3. Iterate through panels and generate tasks with context
        print(f"🎨 Generating {len(all_panels)} panels (from {len(script_data)} pages) in parallel...")
        tasks = []
        
        for i, item in enumerate(all_panels):
            page_num = item['page_num']
            panel = item['panel_data']
            
            # Determine Previous Context
            prev_context = None
            if i > 0:
                prev_item = all_panels[i-1]
                prev_desc = prev_item['panel_data'].get('visual_description', '')
                prev_page = prev_item['page_num']
                # Include page boundary info if applicable
                context_prefix = ""
                if prev_page != page_num:
                    context_prefix = "[PREVIOUS PAGE FINAL PANEL] "
                prev_context = f"{context_prefix}{prev_desc}"

            # Determine Next Context
            next_context = None
            if i < len(all_panels) - 1:
                next_item = all_panels[i+1]
                next_desc = next_item['panel_data'].get('visual_description', '')
                next_page = next_item['page_num']
                # Include page boundary info if applicable
                context_prefix = ""
                if next_page != page_num:
                    context_prefix = "[NEXT PAGE FIRST PANEL] "
                next_context = f"{context_prefix}{next_desc}"

            tasks.append(self.generate_panel(page_num, panel, prev_context, next_context))
        
        await asyncio.gather(*tasks)

        print(f"\n✅ Production Complete! All panels generated.")

if __name__ == "__main__":
    # Configuration
    # Point to your generated script JSON
    SCRIPT_FILE = "assets/output/20-thousand-leagues-under-the-sea_test_page.json"
    
    # Define the overarching style. This should match what you used in the Scripting Agent.
    # Be Descriptive! This is appended to every single panel prompt.
    GLOBAL_STYLE = "Lush Watercolor comic book art. Dreamlike quality, soft color bleeds, visible paper texture. Ethereal lighting."

    agent = IllustratorAgent(SCRIPT_FILE, GLOBAL_STYLE)
    asyncio.run(agent.run_production())