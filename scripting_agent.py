import os
import json
import asyncio
import hashlib
from pathlib import Path
from dataclasses import dataclass, field
from dotenv import load_dotenv

# Load .env BEFORE importing config
load_dotenv()

from google import genai
from google.genai import types
from utils import retry_with_backoff, RateLimiter
from config import config

# Initialize Client
client = genai.Client(api_key=config.gemini_api_key)

# Rate limiter for the parallel script writing phase
# We can go higher here because we aren't generating images yet
scribe_limiter = RateLimiter(rpm_limit=15)

class ScriptingAgent:
    def __init__(self, book_path: str):
        self.book_path = Path(book_path)
        self.output_dir = config.output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cache_name = None

    def load_content(self, test_mode=True):
        """
        Loads the book content. 
        """
        with open(self.book_path, "r", encoding="utf-8") as f:
            full_text = f.read()
            if test_mode:
                # Take a generous slice for testing
                return full_text[:50000] 
            return full_text

    async def _get_or_create_cache(self, content: str):
        """
        Uploads the book content to Gemini's Context Cache.
        Returns the cache object or name to reference in subsequent calls.
        """
        # Create a deterministic hash for the cache name based on content
        content_hash = hashlib.md5(content.encode('utf-8')).hexdigest()
        # Cache names must be unique to the project/content
        # In a real app, we might check if a cache with this name already exists via list_caches()
        # For this prototype, we'll create a new one with a TTL. 
        
        print(f"💾 Caching book content ({len(content)} chars) to Gemini Context Cache...")
        
        try:
            # Note: The SDK for caching might vary slightly by version. 
            # We are using the 'google.genai' library.
            
            # 1. Create the cache
            # We use the 'scripting_model_global_context' as the base model for the cache
            # or a known caching-capable model like gemini-1.5-pro-002
            
            # Using a distinct display name helps debugging
            display_name = f"book_cache_{self.book_path.stem}_{content_hash[:6]}"
            
            cached_content = await client.aio.caches.create(
                model=config.scripting_model_global_context,
                config=types.CreateCachedContentConfig(
                    display_name=display_name,
                    contents=[content],
                    ttl="3600s" # 1 hour TTL
                )
            )
            
            self.cache_name = cached_content.name
            print(f"✅ Cache created: {self.cache_name} (TTL: 1h)")
            return self.cache_name

        except Exception as e:
            print(f"⚠️ Cache creation failed: {e}")
            print("   Falling back to sending full text in context (slower/more expensive).")
            return None

    @retry_with_backoff()
    async def analyze_global_context(self, full_text: str):
        """
        DEPRECATED: Retained for compatibility with current production_run.py.
        The 'Director' pass now handles this implicitly.
        """
        print("ℹ️  (Legacy) Global Context Analysis merged into Director pass.")
        return "Standard Graphic Novel Adaptation"

    @retry_with_backoff()
    async def generate_pacing_blueprint(self, cache_name: str, full_text_fallback: str, target_pages: int, style: str):
        """
        PASS 1: THE DIRECTOR
        Consumes the FULL BOOK (via cache) and outputs a page-by-page blueprint.
        """
        print(f"🎬 DIRECTOR PASS: Creating {target_pages}-page blueprint...")
        
        prompt = f"""
        Act as a Master Graphic Novel Director.
        
        TASK:
        Adapt the provided book into a TIGHT {target_pages}-PAGE Graphic Novel Script.
        
        You must output a JSON list of exactly {target_pages} items. 
        Each item represents ONE PAGE and must define:
        1. 'page_number': Integer (1 to {target_pages}).
        2. 'summary': A 2-sentence summary of what happens on this page.
        3. 'focus_text': A specific quote or 200-word excerpt from the source text that this page covers.
        4. 'mood': The emotional tone (e.g., "Tense", "Melancholic").
        5. 'key_characters': List of characters present.
        6. 'visual_notes': Specific setting or lighting notes.

        CRITICAL PACING RULES:
        - Page 1 MUST introduce the setting/protagonist.
        - Page {target_pages} MUST contain the ending or a major cliffhanger.
        - Distribute the story arc evenly. Do not rush the ending.
        
        STYLE: {style}
        
        OUTPUT FORMAT: JSON List.
        """

        # Prepare request arguments
        model = config.scripting_model_global_context
        
        if cache_name:
            # Use cached content
            contents = [prompt]
            cached_content = cache_name
        else:
            # Fallback: Send full text (truncate if absolutely massive)
            contents = [prompt, f"SOURCE BOOK:\n{full_text_fallback}"]
            cached_content = None

        response = await client.aio.models.generate_content(
            model=model,
            contents=contents,
            config=types.GenerateContentConfig(
                cached_content=cached_content, # Pass the cache name here
                response_mime_type="application/json",
                response_schema={
                    "type": "ARRAY",
                    "items": {
                        "type": "OBJECT",
                        "properties": {
                            "page_number": {"type": "INTEGER"},
                            "summary": {"type": "STRING"},
                            "focus_text": {"type": "STRING"},
                            "mood": {"type": "STRING"},
                            "key_characters": {"type": "ARRAY", "items": {"type": "STRING"}},
                            "visual_notes": {"type": "STRING"}
                        },
                        "required": ["page_number", "summary", "focus_text", "key_characters"]
                    }
                }
            )
        )
        
        return response.parsed

    @retry_with_backoff()
    async def write_page_script(self, cache_name: str, full_text_fallback: str, blueprint_item: dict, style: str):
        """
        PASS 2: THE SCRIPTWRITER
        Generates the detailed panel script for a SINGLE page, strictly following the blueprint.
        """
        page_num = blueprint_item['page_number']
        
        async with scribe_limiter:
            print(f"✍️  Scripting Page {page_num}...")
            
            prompt = f"""
            Act as a Graphic Novel Scriptwriter.
            
            TASK:
            Write the panel-by-panel script for PAGE {page_num}.
            
            BLUEPRINT FOR THIS PAGE:
            Summary: {blueprint_item['summary']}
            Mood: {blueprint_item['mood']}
            Visual Notes: {blueprint_item['visual_notes']}
            Focus Text/Context: "{blueprint_item['focus_text']}"
            
            STYLE: {style}
            
            INSTRUCTIONS:
            1. Break this page into 3-6 panels.
            2. Use 'visual_description' for the artist (cinematic, detailed).
            3. Use 'dialogue' for characters (keep it brief, punchy).
            4. Use 'caption' for narration.
            5. Ensure visual continuity with the blueprint notes.
            
            OUTPUT FORMAT: JSON.
            """

            # Prepare request
            model = config.scripting_model_page_script
            
            if cache_name:
                contents = [prompt]
                cached_content = cache_name
            else:
                # For specific page scripting without cache, we ideally need context. 
                # Since we are in fallback mode, we pass the 'focus_text' from blueprint 
                # plus a bit of the summary as the "source".
                # We do NOT send the full book again to save tokens if cache failed.
                contents = [prompt, f"RELEVANT SOURCE TEXT:\n{blueprint_item['focus_text']}"]
                cached_content = None

            response = await client.aio.models.generate_content(
                model=model,
                contents=contents,
                config=types.GenerateContentConfig(
                    cached_content=cached_content,
                    response_mime_type="application/json",
                    response_schema={
                        "type": "OBJECT",
                        "properties": {
                            "page_number": {"type": "INTEGER"},
                            "panels": {
                                "type": "ARRAY",
                                "items": {
                                    "type": "OBJECT",
                                    "properties": {
                                        "panel_id": {"type": "INTEGER"},
                                        "visual_description": {"type": "STRING"},
                                        "dialogue": {"type": "STRING"},
                                        "caption": {"type": "STRING"},
                                        "characters": {"type": "ARRAY", "items": {"type": "STRING"}},
                                        "bubble_position": {"type": "STRING", "enum": ["top-left", "top-right", "bottom-left", "bottom-right", "caption-box"]}
                                    },
                                    "required": ["panel_id", "visual_description", "characters"]
                                }
                            }
                        },
                        "required": ["page_number", "panels"]
                    }
                )
            )
            
            # Post-process: Ensure page number matches
            result = response.parsed
            result['page_number'] = page_num
            return result

    async def generate_script(self, style: str, test_mode=True, context_constraints: str = "", target_page_override: int = None):
        """
        Main orchestration method:
        1. Load Content
        2. Create Cache
        3. Director Pass (Blueprint)
        4. Scriptwriter Pass (Parallel Execution)
        """
        # Output path
        suffix = "_test_page.json" if test_mode else "_full_script.json"
        output_file = self.output_dir / f"{self.book_path.stem}{suffix}"
        
        if output_file.exists() and not test_mode:
             print(f"⏭️  Skipping Script Generation (Already exists: {output_file})")
             with open(output_file, "r") as f:
                 return json.load(f)

        # 1. Load Content
        full_text = self.load_content(test_mode=False) # Always load full for caching
        target_pages = target_page_override or (3 if test_mode else 10)
        
        print(f"\n📚 Scripting '{self.book_path.stem}'")
        print(f"   Target Length: {target_pages} pages")
        print(f"   Style: {style}")

        # 2. Create Cache (The Enabler)
        cache_name = await self._get_or_create_cache(full_text)

        # 3. Director Pass (Blueprint)
        blueprint = await self.generate_pacing_blueprint(cache_name, full_text, target_pages, style)
        
        # Save blueprint for debugging
        blueprint_path = self.output_dir / f"{self.book_path.stem}_blueprint.json"
        with open(blueprint_path, "w") as f:
            json.dump(blueprint, f, indent=2)
        print(f"✅ Blueprint created: {blueprint_path}")

        # 4. Scriptwriter Pass (Parallel)
        print(f"⚡ Starting Parallel Script Generation for {len(blueprint)} pages...")
        
        tasks = [
            self.write_page_script(cache_name, full_text, item, style)
            for item in blueprint
        ]
        
        # Execute all pages at once (bounded by scribe_limiter)
        full_script = await asyncio.gather(*tasks)
        
        # Sort by page number just in case
        full_script.sort(key=lambda x: x['page_number'])

        # Save Final Script
        with open(output_file, "w") as f:
            json.dump(full_script, f, indent=2)

        print(f"🎉 Script Generation Complete! Saved to: {output_file}")
        return full_script

if __name__ == "__main__":
    # Test run
    agent = ScriptingAgent("assets/input/20-thousand-leagues-under-the-sea.txt")
    asyncio.run(agent.generate_script(
        style="Lush Watercolor", 
        test_mode=True, 
        target_page_override=3
    ))