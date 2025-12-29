import os
import json
import asyncio
from pathlib import Path
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

class ScriptingAgent:
    def __init__(self, book_path: str):
        self.book_path = Path(book_path)
        self.output_dir = Path("assets/output")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_content(self, test_mode=True):
        """
        Loads the book content. 
        In Test Mode: Returns a 10k char slice to save tokens/time.
        In Prod Mode: Returns the full text.
        """
        with open(self.book_path, "r", encoding="utf-8") as f:
            full_text = f.read()
            if test_mode:
                # Take a slice from the middle
                mid = len(full_text) // 2
                start = max(0, mid - 5000)
                end = min(len(full_text), mid + 5000)
                return f"...{full_text[start:end]}..."
            return full_text

    async def generate_beat_sheet(self, source_text: str, style: str, target_page_count: int = 30):
        """
        THE ARCHITECT: break the story down into a page-by-page outline.
        This ensures the pacing covers the WHOLE story, not just the first chapter.
        """
        print(f"🏗️  Architecting story structure (Target: {target_page_count} pages)...")
        
        prompt = f"""
        Act as a Master Editor adapting a novel into a {target_page_count}-page Graphic Novel.
        Your goal is ADAPTATION, not translation. You must condense the story while preserving the narrative arc, emotional beats, and key action.

        STYLE: {style}
        
        TASK:
        Create a numbered Beat Sheet for exactly {target_page_count} pages.
        For each page, provide:
        1. 'page_number': The integer page number.
        2. 'narrative_goal': What plot point happens on this page? (e.g., "The Nautilus is revealed", "Escape attempt fails").
        3. 'key_visual': The most important visual anchor for this page.
        4. 'atmosphere': The mood of this specific page.

        Ensure the story has a beginning, middle, and end within these {target_page_count} pages.
        
        OUTPUT FORMAT: JSON List of Objects.
        """
        
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[prompt, source_text],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema={
                    "type": "ARRAY",
                    "items": {
                        "type": "OBJECT",
                        "properties": {
                            "page_number": {"type": "INTEGER"},
                            "narrative_goal": {"type": "STRING"},
                            "key_visual": {"type": "STRING"},
                            "atmosphere": {"type": "STRING"}
                        },
                        "required": ["page_number", "narrative_goal"]
                    }
                }
            )
        )
        
        # Save Beat Sheet for Review
        beat_sheet_path = self.output_dir / f"{self.book_path.stem}_beat_sheet.json"
        with open(beat_sheet_path, "w") as f:
            json.dump(response.parsed, f, indent=2)
            
        print(f"✅ Beat Sheet Saved: {beat_sheet_path}")
        return response.parsed

    async def write_page_script(self, beat: dict, source_text: str, style: str, tone: str):
        """
        THE SCRIBE: Writes the detailed panel directions for a SINGLE page based on the Beat Sheet.
        """
        print(f"✍️  Scripting Page {beat['page_number']}: {beat['narrative_goal']}...")
        
        prompt = f"""
        Act as a Graphic Novel Scriptwriter.
        
        CONTEXT:
        We are adapting a novel into a graphic novel.
        Current Page Goal: {beat['narrative_goal']}
        Key Visual: {beat['key_visual']}
        Atmosphere: {beat['atmosphere']}
        Style: {style}
        Tone: {tone}
        
        INSTRUCTIONS:
        Write the script for Page {beat['page_number']}.
        - Break this specific scene into 1-6 panels.
        - Ensure a logical visual flow (Z-pattern).
        - VISUAL DESCRIPTION: Must be highly descriptive for an AI image generator (mention lighting, angles, colors).
        - DIALOGUE: Keep it punchy. Use standard comic conventions.
        - BUBBLE POSITION: Ensure text doesn't cover faces.
        
        Reference the Source Text provided to maintain character voice, but feel free to abridge dialogue significantly.
        """

        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[prompt, "SOURCE TEXT CONTEXT:\n" + source_text],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema={
                    "type": "OBJECT",
                    "properties": {
                        "page_number": {"type": "INTEGER"},
                        "layout_style": {"type": "STRING", "enum": ["cinematic_widescreen", "action_dynamic", "standard_grid", "dialogue_focus", "full_splash"]},
                        "panels": {
                            "type": "ARRAY",
                            "items": {
                                "type": "OBJECT",
                                "properties": {
                                    "panel_id": {"type": "INTEGER"},
                                    "visual_description": {"type": "STRING"},
                                    "dialogue": {"type": "STRING"},
                                    "characters": {"type": "ARRAY", "items": {"type": "STRING"}},
                                    "bubble_position": {"type": "STRING", "enum": ["top-left", "top-right", "bottom-left", "bottom-right"]}
                                },
                                "required": ["panel_id", "visual_description", "dialogue", "characters", "bubble_position"]
                            }
                        }
                    },
                    "required": ["page_number", "panels"]
                }
            )
        )
        return response.parsed

    async def generate_script(self, style: str, tone: str, writing_style: str, test_mode=True):
        source_text = self.load_content(test_mode=test_mode)
        
        # 1. THE ARCHITECT PHASE
        # If test mode, we just do 1 page. If full, we do 20 pages (for this demo, to save time/cost).
        target_pages = 1 if test_mode else 10
        beat_sheet = await self.generate_beat_sheet(source_text, style, target_pages)
        
        # 2. THE SCRIBE PHASE
        full_script = []
        
        # In a real heavy-duty app, we might parallelize this with asyncio.gather, 
        # but serial is safer for rate limits and debugging.
        for beat in beat_sheet:
            page_script = await self.write_page_script(beat, source_text, style, tone)
            full_script.append(page_script)

        # Save the result
        suffix = "_test_page.json" if test_mode else "_full_script.json"
        output_file = self.output_dir / f"{self.book_path.stem}{suffix}"
        
        with open(output_file, "w") as f:
            json.dump(full_script, f, indent=2)
        
        print(f"✅ Script Generation Complete! Saved to: {output_file}")
        return full_script

if __name__ == "__main__":
    agent = ScriptingAgent("assets/input/20-thousand-leagues-under-the-sea.txt")
    asyncio.run(agent.generate_script(style="Watercolor", tone="Melancholic", writing_style="Cinematic", test_mode=True))

