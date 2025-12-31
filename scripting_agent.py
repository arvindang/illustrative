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
scribe_limiter = RateLimiter(rpm_limit=5)

# Model configuration for different tasks
# Note: Using gemini-3-flash-preview as it's currently available
# TODO: Update to gemini-2.0-flash-exp or gemini-2.0-flash when stable API is available
MODEL_CONFIG = {
    "global_context": "gemini-2.5-pro",      # Structured output, straightforward
    "chapter_map": "gemini-3-flash-preview",         # Structured output
    "beat_sheet": "gemini-2.5-pro",          # Story architecture
    "page_script": "gemini-3-flash-preview",         # High volume, structured
}

class ScriptingAgent:
    def __init__(self, book_path: str):
        self.book_path = Path(book_path)
        self.output_dir = Path("assets/output")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def get_model_for_task(self, task_type: str) -> str:
        """Select appropriate model based on task requirements."""
        return MODEL_CONFIG.get(task_type, "gemini-2.0-flash")

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
    async def analyze_global_context(self, full_text: str):
        """
        Act as a Production Designer. Analyze the text to determine the 
        setting, era, technology level, and specific visual requirements.
        """
        context_file = self.output_dir / f"{self.book_path.stem}_context.txt"
        if context_file.exists():
            print("⏭️  Skipping Global Context Analysis (Already exists)")
            with open(context_file, "r") as f:
                return f.read().strip()

        print("🔍 Analyzing Global Context & Historical Constraints...")
        prompt = """
        Act as a Production Designer and Historical Consultant for a Graphic Novel adaptation.
        Analyze the provided text and extract a concise 'Production Bible' of constraints.
        
        Focus on:
        1. Setting/Era (e.g., 1860s Victorian).
        2. Technology level (e.g., Steam-powered, specific gear mentioned).
        3. Visual Constants (e.g., "Nautilus must look like a sea monster", "Diving suits are copper").
        4. Atmospheric details.

        OUTPUT: A single string (2-4 sentences) that summarizes these constraints for a creative team.
        """
        
        response = await client.aio.models.generate_content(
            model=self.get_model_for_task("global_context"),
            contents=[prompt, full_text[:30000]], # First 30k chars usually enough for setting
        )
        
        context = response.text.strip()
        with open(context_file, "w") as f:
            f.write(context)
        return context

    @retry_with_backoff()
    async def generate_chapter_map(self, full_text: str):
        """
        Creates a high-level map of the book to help with contextual slicing.
        """
        chapter_map_path = self.output_dir / f"{self.book_path.stem}_chapter_map.json"
        if chapter_map_path.exists():
            print("⏭️  Skipping Chapter Map Generation (Already exists)")
            with open(chapter_map_path, "r") as f:
                return json.load(f)

        print("🗺️  Generating Chapter Index (simplified)...")
        prompt = """
        Act as a Book Indexer. Analyze the provided book text.
        Break the book down into logical chapters or segments.
        For each segment, provide:
        1. 'chapter_number': Integer.
        2. 'title': Short descriptive title for this chapter.
        3. 'start_phrase': The first 10-15 words of the chapter (for text slicing).
        4. 'end_phrase': The last 10-15 words of the chapter (for text slicing).
        5. 'estimated_word_count': Approximate word count for this chapter.

        This is a STRUCTURAL INDEX only - do NOT analyze narrative beats or extract dialogue.
        Keep this lightweight and fast.

        OUTPUT FORMAT: JSON List of Objects.
        """
        
        response = await client.aio.models.generate_content(
            model=self.get_model_for_task("chapter_map"),
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
                            "start_phrase": {"type": "STRING"},
                            "end_phrase": {"type": "STRING"},
                            "estimated_word_count": {"type": "INTEGER"}
                        },
                        "required": ["chapter_number", "title", "start_phrase", "end_phrase"]
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

    def calculate_beat_density(self, chapter_map: list, test_mode: bool = False) -> dict:
        """
        Calculate page allocation guidance based on narrative beat density.
        Integrates with existing word-count-based page calculation system.

        Returns:
            Dict with total_beats, pages_per_beat ratio, target_pages, and per-chapter guidance
        """
        total_beats = sum(len(ch.get('key_beats', [])) for ch in chapter_map)

        # Use existing dynamic page count system
        from utils import calculate_page_count
        word_count = len(self.load_content(test_mode=False).split())
        page_calc = calculate_page_count(word_count, test_mode=test_mode)
        target_pages = page_calc['recommended']

        # Calculate pages per beat ratio
        pages_per_beat = target_pages / total_beats if total_beats > 0 else 1.0

        # Generate per-chapter guidance
        chapter_guidance = []
        for ch in chapter_map:
            beat_count = len(ch.get('key_beats', []))
            suggested_pages = max(1, int(beat_count * pages_per_beat))
            chapter_guidance.append({
                'chapter_number': ch.get('chapter_number', 0),
                'title': ch.get('title', 'Untitled'),
                'beat_count': beat_count,
                'suggested_pages': suggested_pages
            })

        return {
            'total_beats': total_beats,
            'target_pages': target_pages,
            'pages_per_beat': pages_per_beat,
            'chapter_guidance': chapter_guidance,
            'word_count': word_count
        }

    def get_chapter_text_by_phrase(self, full_text: str, start_phrase: str, end_phrase: str) -> str:
        """
        Extract chapter text using start and end phrases.
        More reliable than line number indexing.
        """
        start_idx = full_text.find(start_phrase)
        end_idx = full_text.find(end_phrase, start_idx)

        if start_idx == -1 or end_idx == -1:
            # Fallback: return empty or raise error
            print(f"⚠️ Could not locate chapter boundaries")
            return ""

        # Include the end phrase
        end_idx += len(end_phrase)

        return full_text[start_idx:end_idx]

    def allocate_pages_simple(self, chapter_index: list, target_page_count: int) -> list:
        """
        Simple page allocation based on word count distribution.
        Each chapter gets pages proportional to its word count.

        Returns: List of page allocations with chapter assignments
        """
        total_words = sum(ch.get('estimated_word_count', 1000) for ch in chapter_index)
        words_per_page = total_words / target_page_count

        print(f"📊 Page Allocation: {total_words:,} words ÷ {target_page_count} pages = {words_per_page:.0f} words/page")

        page_allocations = []
        current_page = 1

        for chapter in chapter_index:
            chapter_words = chapter.get('estimated_word_count', 1000)
            chapter_pages = max(1, round(chapter_words / words_per_page))

            # Assign pages to this chapter
            for i in range(chapter_pages):
                if current_page > target_page_count:
                    break

                page_allocations.append({
                    'page_number': current_page,
                    'chapter_number': chapter['chapter_number'],
                    'chapter_title': chapter['title'],
                    'page_within_chapter': i + 1,
                    'total_pages_in_chapter': chapter_pages
                })
                current_page += 1

            if current_page > target_page_count:
                break

        # Trim to exact target (in case rounding gave us extra)
        page_allocations = page_allocations[:target_page_count]

        print(f"✅ Allocated {len(page_allocations)} pages across {len(chapter_index)} chapters")
        return page_allocations

    @retry_with_backoff()
    async def generate_beat_sheet(self, source_text: str, chapter_map: list, style: str, target_page_count: int = 30):
        """
        THE ARCHITECT: break the story down into a page-by-page outline.
        Uses the Chapter Map to ensure global coverage.
        """
        beat_sheet_path = self.output_dir / f"{self.book_path.stem}_beat_sheet.json"
        if beat_sheet_path.exists():
            # Check if existing beat sheet matches the target page count
            with open(beat_sheet_path, "r") as f:
                existing_beat_sheet = json.load(f)
                if len(existing_beat_sheet) == target_page_count:
                    print(f"⏭️  Skipping Beat Sheet Generation (Already exists with {target_page_count} pages)")
                    return existing_beat_sheet

        print(f"🏗️  Architecting story structure (Target: {target_page_count} pages)...")

        # Calculate beat density for intelligent page allocation
        beat_density = self.calculate_beat_density(chapter_map, test_mode=(target_page_count <= 10))

        chapter_summary = json.dumps(chapter_map, indent=2)
        chapter_guidance_summary = json.dumps(beat_density['chapter_guidance'], indent=2)

        prompt = f"""
        Act as a Master Editor adapting a novel into a {target_page_count}-page Graphic Novel.
        Your goal is ADAPTATION, not translation. You must condense the story while preserving the narrative arc, emotional beats, and key action.

        STYLE: {style}

        CHAPTER NARRATIVE SKELETON:
        {chapter_summary}

        BEAT DENSITY ANALYSIS:
        - Total narrative beats across all chapters: {beat_density['total_beats']}
        - Target page count (based on {beat_density['word_count']} words): {beat_density['target_pages']}
        - Ratio: {beat_density['pages_per_beat']:.3f} pages per beat

        CHAPTER PAGE ALLOCATION GUIDANCE:
        {chapter_guidance_summary}

        TASK:
        Create a numbered Beat Sheet for exactly {target_page_count} pages.

        INSTRUCTIONS FOR PAGE ALLOCATION:
        - Use the key_beats arrays to understand narrative density of each chapter
        - Follow the suggested_pages guidance above as a starting point
        - You may adjust ±20% based on:
          1. Narrative importance (climax chapters may deserve more pages)
          2. Visual complexity (action scenes vs dialogue)
          3. Story pacing (slow buildup vs fast resolution)
        - Allocate exactly {target_page_count} pages total

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
            model=self.get_model_for_task("beat_sheet"),
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
    async def write_page_script(
        self,
        page_allocation: dict,  # Simple page allocation with chapter info
        full_text: str,         # FULL book text
        chapter_index: list,    # Minimal structural index
        style: str,
        tone: str,
        context_constraints: str = "",
        dialogue_mode: str = "balanced"
    ):
        """
        THE SCRIBE: Writes the detailed panel directions for a SINGLE page with FULL TEXT ACCESS.
        """
        async with scribe_limiter:
            # Get chapter info
            chapter_num = page_allocation['chapter_number']
            chapter = next(ch for ch in chapter_index if ch['chapter_number'] == chapter_num)

            print(f"✍️  Scripting Page {page_allocation['page_number']}: Chapter {chapter_num} ({chapter['title']}) - Page {page_allocation['page_within_chapter']}/{page_allocation['total_pages_in_chapter']}...")

            # Extract full chapter text using start/end phrases
            chapter_text = self.get_chapter_text_by_phrase(full_text, chapter['start_phrase'], chapter['end_phrase'])

            # Also include adjacent chapters for context (prevent boundary issues)
            prev_chapter_text = ""
            next_chapter_text = ""

            if chapter_num > 1:
                prev_ch = next((ch for ch in chapter_index if ch['chapter_number'] == chapter_num - 1), None)
                if prev_ch:
                    prev_full = self.get_chapter_text_by_phrase(full_text, prev_ch['start_phrase'], prev_ch['end_phrase'])
                    prev_chapter_text = prev_full[-2000:] if len(prev_full) > 2000 else prev_full  # Last 2000 chars

            if chapter_num < len(chapter_index):
                next_ch = next((ch for ch in chapter_index if ch['chapter_number'] == chapter_num + 1), None)
                if next_ch:
                    next_full = self.get_chapter_text_by_phrase(full_text, next_ch['start_phrase'], next_ch['end_phrase'])
                    next_chapter_text = next_full[:2000] if len(next_full) > 2000 else next_full  # First 2000 chars

            # Build context
            full_context = f"""
PREVIOUS CHAPTER (for context):
{prev_chapter_text}

CURRENT CHAPTER TEXT:
{chapter_text}

NEXT CHAPTER (for context):
{next_chapter_text}
"""

            prompt = f"""
Act as a Graphic Novel Scriptwriter.

TASK:
You are adapting Chapter {chapter_num}: "{chapter['title']}" into a graphic novel.
This is Page {page_allocation['page_number']} of the entire adaptation.
This is Page {page_allocation['page_within_chapter']} of {page_allocation['total_pages_in_chapter']} pages covering this chapter.

Style: {style}
Tone: {tone}

GLOBAL CONTEXT & CONSTRAINTS:
{context_constraints}

INSTRUCTIONS:
Write the script for Page {page_allocation['page_number']}.
- Create 1-6 panels that capture the narrative for this portion of the chapter
- If this is page 1 of the chapter, start with the chapter opening
- If this is the last page of the chapter, conclude with the chapter ending
- VISUAL DESCRIPTION: Highly descriptive for AI image generation (lighting, angles, colors)
- DIALOGUE: PRESERVE character voice and emotional depth from the source text
  * Adapt actual dialogue directly from the chapter text
  * Maintain character speech patterns and personality
  * Show, don't tell - use visuals to convey prose descriptions
  CRITICAL: Do NOT include character names like "Narrator:" or "Nemo:" in the dialogue string. ONLY provide the text to be spoken or narrated.
- BUBBLE POSITION: Ensure text doesn't cover faces
- CHARACTERS: List specific sentient characters present (no objects/animals unless primary)
  * DO NOT include objects (e.g., "Nautilus", "Ship", "Harpoon").
  * DO NOT include animals unless they are primary characters.
  * If no characters are present, return an empty list [].
  * DO NOT use placeholders like "none" or "no characters".
- ADVICE: Provide structured guidance:
  * continuity_notes: Props, clothing, positions from previous panels
  * historical_constraints: Era-specific requirements
  * character_gear: What characters are holding, wearing

CRITICAL: The FULL CHAPTER TEXT is provided below. Read it carefully and adapt dialogue
and scenes directly from it. Do NOT synthesize or invent dialogue - use the author's actual words.

OUTPUT FORMAT: JSON with panels array.
"""

            models_to_try = [self.get_model_for_task("page_script"), "gemini-2.0-flash"]
            last_error = None

            for model_name in models_to_try:
                try:
                    response = await client.aio.models.generate_content(
                        model=model_name,
                        contents=[prompt, f"SOURCE TEXT:\n{full_context}"],
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
                                                "advice": {
                                                    "type": "OBJECT",
                                                    "properties": {
                                                        "continuity_notes": {"type": "STRING"},
                                                        "historical_constraints": {"type": "STRING"},
                                                        "character_gear": {"type": "STRING"}
                                                    },
                                                    "required": ["continuity_notes", "historical_constraints", "character_gear"]
                                                },
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
                except Exception as e:
                    last_error = e
                    if "429" in str(e):
                        print(f"⚠️ Model {model_name} rate limited on Page {beat['page_number']}. Trying fallback...")
                        continue
                    raise e
            
            raise last_error

    async def generate_script(self, style: str, tone: str, writing_style: str, test_mode=True, context_constraints: str = "", target_page_override: int = None):
        # Save the result
        suffix = "_test_page.json" if test_mode else "_full_script.json"
        output_file = self.output_dir / f"{self.book_path.stem}{suffix}"
        
        if output_file.exists():
            print(f"⏭️  Skipping Full Script Generation (Already exists: {output_file})")
            with open(output_file, "r") as f:
                return json.load(f)

        full_text = self.load_content(test_mode=False) # Always load full for map

        # Calculate optimal page count
        from utils import calculate_page_count
        word_count = len(full_text.split())

        page_calc = calculate_page_count(
            word_count=word_count,
            test_mode=test_mode,
            user_override=target_page_override
        )

        # Display calculation
        print(f"\n📊 Input Analysis:")
        print(f"   Words: {word_count:,}")
        print(f"   Category: {page_calc['density_category']}")
        print(f"   Target pages: {page_calc['recommended']} (range: {page_calc['minimum']}-{page_calc['maximum']})")
        print(f"   Est. time: ~{page_calc['estimated_time_minutes']} min")
        if page_calc['warning']:
            print(f"   ⚠️  {page_calc['warning']}\n")

        target_pages = page_calc['recommended']

        # 1. THE MAPPER PHASE
        # We generate a chapter map to help slice the story
        chapter_map = await self.generate_chapter_map(full_text)

        # 2. THE ARCHITECT PHASE
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

