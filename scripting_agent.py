import os
import json
import asyncio
from pathlib import Path
from dotenv import load_dotenv
from google import genai
from google.genai import types
from utils import retry_with_backoff, RateLimiter

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
# Using a semaphore to throttle the scribe phase
scribe_limiter = RateLimiter(rpm_limit=10)

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

    @retry_with_backoff()
    async def generate_chapter_map(self, full_text: str):
        """
        Creates a high-level map of the book to help with contextual slicing.
        """
        print("🗺️  Generating Chapter Map (Gemini 2.5 Pro)...")
        prompt = """
        Act as a Literary Analyst. Analyze the provided book text.
        Break the book down into logical chapters or segments.
        For each segment, provide:
        1. 'chapter_number': Integer.
        2. 'title': Short title.
        3. 'summary': 2-3 sentence summary of key events.
        4. 'main_characters': List of characters appearing.
        5. 'start_phrase': The first 10 words of the chapter (to help locating it).
        6. 'end_phrase': The last 10 words of the chapter.

        OUTPUT FORMAT: JSON List of Objects.
        """
        
        response = await client.aio.models.generate_content(
            model="gemini-2.5-pro", # Use Pro for high-context analysis
            contents=[prompt, full_text],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema={
                    "type": "ARRAY",
                    "items": {
                        "type": "OBJECT",
                        "properties": {
                            "chapter_number": {"type": "INTEGER"},
                            "title": {"type": "STRING"},
                            "summary": {"type": "STRING"},
                            "main_characters": {"type": "ARRAY", "items": {"type": "STRING"}},
                            "start_phrase": {"type": "STRING"},
                            "end_phrase": {"type": "STRING"}
                        },
                        "required": ["chapter_number", "title", "summary"]
                    }
                }
            )
        )
        
        chapter_map_path = self.output_dir / f"{self.book_path.stem}_chapter_map.json"
        with open(chapter_map_path, "w") as f:
            json.dump(response.parsed, f, indent=2)
            
        print(f"✅ Chapter Map Saved: {chapter_map_path}")
        return response.parsed

    def get_chapter_text(self, full_text: str, chapter_map: list, chapter_numbers: list):
        """
        Slices the full text to include only the requested chapters.
        """
        if not chapter_numbers:
            return full_text[:20000] # Fallback
            
        context_parts = []
        for num in chapter_numbers:
            # Find chapter in map
            chap = next((c for c in chapter_map if c['chapter_number'] == num), None)
            if not chap:
                continue
                
            start_phrase = chap.get('start_phrase', '')
            end_phrase = chap.get('end_phrase', '')
            
            try:
                start_idx = full_text.find(start_phrase) if start_phrase else 0
                if start_idx == -1: start_idx = 0
                
                end_idx = full_text.find(end_phrase, start_idx) if end_phrase else len(full_text)
                if end_idx == -1: end_idx = len(full_text)
                else: end_idx += len(end_phrase)
                
                context_parts.append(full_text[start_idx:end_idx])
            except Exception:
                continue
                
        return "\n\n--- NEXT CHAPTER ---\n\n".join(context_parts) if context_parts else full_text[:20000]

    @retry_with_backoff()
    async def generate_beat_sheet(self, source_text: str, chapter_map: list, style: str, target_page_count: int = 30):
        """
        THE ARCHITECT: break the story down into a page-by-page outline.
        Uses the Chapter Map to ensure global coverage.
        """
        print(f"🏗️  Architecting story structure (Target: {target_page_count} pages)...")
        
        chapter_summary = json.dumps(chapter_map, indent=2)
        
        prompt = f"""
        Act as a Master Editor adapting a novel into a {target_page_count}-page Graphic Novel.
        Your goal is ADAPTATION, not translation. You must condense the story while preserving the narrative arc, emotional beats, and key action.

        STYLE: {style}
        
        CHAPTER MAP:
        {chapter_summary}

        TASK:
        Create a numbered Beat Sheet for exactly {target_page_count} pages.
        For each page, provide:
        1. 'page_number': The integer page number (MUST be sequential starting from 1).
        2. 'narrative_goal': What plot point happens on this page?
        3. 'key_visual': The most important visual anchor for this page.
        4. 'atmosphere': The mood of this specific page.
        5. 'relevant_chapters': A list of chapter numbers from the map that this page covers.

        Ensure the story has a beginning, middle, and end within these {target_page_count} pages.
        
        OUTPUT FORMAT: JSON List of Objects.
        """
        
        response = await client.aio.models.generate_content(
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
                            "atmosphere": {"type": "STRING"},
                            "relevant_chapters": {"type": "ARRAY", "items": {"type": "INTEGER"}}
                        },
                        "required": ["page_number", "narrative_goal", "relevant_chapters"]
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

    @retry_with_backoff()
    async def write_page_script(self, beat: dict, source_text: str, style: str, tone: str, context_constraints: str = ""):
        """
        THE SCRIBE: Writes the detailed panel directions for a SINGLE page based on the Beat Sheet.
        """
        async with scribe_limiter:
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
            
            GLOBAL CONTEXT & CONSTRAINTS:
            {context_constraints}
            
            INSTRUCTIONS:
            Write the script for Page {beat['page_number']}.
            - Break this specific scene into 1-6 panels.
            - Ensure a logical visual flow (Z-pattern).
            - VISUAL DESCRIPTION: Must be highly descriptive for an AI image generator (mention lighting, angles, colors).
            - DIALOGUE: Keep it punchy. Use standard comic conventions. 
              CRITICAL: Do NOT include character names like "Narrator:" or "Nemo:" in the dialogue string. ONLY provide the text to be spoken or narrated.
            - BUBBLE POSITION: Ensure text doesn't cover faces.
            - ADVICE: Provide a brief "advice" string for each panel regarding historical accuracy, character gear, or specific logic (e.g., "Ensure characters have diving suits", "No modern ships").
            
            Reference the Source Text provided to maintain character voice, but feel free to abridge dialogue significantly.
            """

            response = await client.aio.models.generate_content(
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
                                        "advice": {"type": "STRING"},
                                        "characters": {"type": "ARRAY", "items": {"type": "STRING"}},
                                        "bubble_position": {"type": "STRING", "enum": ["top-left", "top-right", "bottom-left", "bottom-right"]}
                                    },
                                    "required": ["panel_id", "visual_description", "dialogue", "advice", "characters", "bubble_position"]
                                }
                            }
                        },
                        "required": ["page_number", "panels"]
                    }
                )
            )
            return response.parsed

    async def generate_script(self, style: str, tone: str, writing_style: str, test_mode=True, context_constraints: str = ""):
        full_text = self.load_content(test_mode=False) # Always load full for map
        
        # 1. THE MAPPER PHASE
        # We generate a chapter map to help slice the story
        chapter_map = await self.generate_chapter_map(full_text)
        
        # 2. THE ARCHITECT PHASE
        # If test mode, we just do 1 page.
        target_pages = 1 if test_mode else 20
        beat_sheet = await self.generate_beat_sheet(full_text, chapter_map, style, target_pages)
        
        # 3. THE SCRIBE PHASE
        print(f"✍️  Scripting {len(beat_sheet)} pages in parallel...")
        
        tasks = []
        for beat in beat_sheet:
            # SLIDING CONTEXT: Only provide text for relevant chapters
            relevant_chapters = beat.get('relevant_chapters', [])
            context_text = self.get_chapter_text(full_text, chapter_map, relevant_chapters)
            
            tasks.append(self.write_page_script(beat, context_text, style, tone, context_constraints))
        
        full_script = await asyncio.gather(*tasks)

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

