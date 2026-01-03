import os
import json
import asyncio
import re
from pathlib import Path
from dataclasses import dataclass, field
from dotenv import load_dotenv

# Load .env BEFORE importing config (which reads env vars at import time)
load_dotenv()

from google import genai
from google.genai import types
from utils import retry_with_backoff, RateLimiter
from config import config


@dataclass
class PageContext:
    """Rolling context passed between sequential page generations."""
    last_panel_summary: str = ""
    character_states: dict = field(default_factory=dict)
    # Format: {"Captain Nemo": {"gear": ["harpoon"], "position": "bridge", "mood": "determined"}}
    open_dialogue_threads: list = field(default_factory=list)
    scene_momentum: str = "establishing"  # establishing | rising | climax | falling | resolution
    # NEW: Emotional thread tracking
    current_emotional_beat: str = "" 
    visual_motifs: list = field(default_factory=list) # e.g., "The growing shadow", "The ticking clock"


# Arc weights for narrative-aware page allocation
ARC_WEIGHTS = {
    "action": 1.3,        # Action-heavy chapters get 30% more pages
    "dialogue": 1.0,      # Standard density
    "contemplative": 0.8, # Reflective chapters slightly compressed
}


def get_position_weight(chapter_num: int, total_chapters: int) -> float:
    """Weight chapters by narrative position (three-act structure)."""
    position = chapter_num / total_chapters
    if position < 0.25:      # Act 1: Setup
        return 0.9
    elif position < 0.75:    # Act 2: Confrontation (includes midpoint)
        if 0.45 < position < 0.55:  # Midpoint
            return 1.2
        return 1.0
    else:                    # Act 3: Resolution
        if position > 0.85:  # Climax zone
            return 1.4
        return 1.1


def extract_context_from_page(page: dict, previous_context: PageContext) -> PageContext:
    """Extract rolling context from a completed page for the next page."""
    panels = page.get('panels', [])
    if not panels:
        return previous_context

    # Get the final panel
    final_panel = panels[-1]

    # Build summary of final panel
    last_panel_summary = f"Panel {final_panel.get('panel_id')}: {final_panel.get('visual_description', '')[:150]}"
    if final_panel.get('dialogue'):
        last_panel_summary += f" | Dialogue: \"{final_panel.get('dialogue')[:80]}...\""

    # Update character states from all panels
    character_states = previous_context.character_states.copy()
    visual_motifs = previous_context.visual_motifs.copy()
    
    for panel in panels:
        advice = panel.get('advice', {})
        gear_info = advice.get('character_gear', '')
        continuity = advice.get('continuity_notes', '')

        for char in panel.get('characters', []):
            if char not in character_states:
                character_states[char] = {"gear": [], "position": "", "mood": ""}
            # Parse gear from advice
            if char.lower() in gear_info.lower():
                character_states[char]["gear_note"] = gear_info
            if char.lower() in continuity.lower():
                character_states[char]["continuity_note"] = continuity
    
    # Update visual motifs if new ones appear (simplified logic)
    # In a real system, we'd have the LLM explicitly output active motifs
    
    # Detect open dialogue threads (questions, interrupted speech)
    open_threads = []
    for panel in panels[-2:]:  # Check last 2 panels
        dialogue = panel.get('dialogue', '')
        if dialogue.endswith('?') or dialogue.endswith('...') or dialogue.endswith('—'):
            chars = panel.get('characters', [])
            speaker = chars[0] if chars else 'Narrator'
            open_threads.append(f"{speaker}: \"{dialogue[-60:]}\"")

    # Determine scene momentum based on panel density and content
    panel_count = len(panels)
    has_action = any('action' in p.get('visual_description', '').lower() for p in panels)
    momentum = "rising" if panel_count >= 4 or has_action else "establishing"

    return PageContext(
        last_panel_summary=last_panel_summary,
        character_states=character_states,
        open_dialogue_threads=open_threads[-3:],  # Keep last 3 threads
        scene_momentum=momentum,
        current_emotional_beat=previous_context.current_emotional_beat, # Persist for now
        visual_motifs=visual_motifs
    )

client = genai.Client(api_key=config.gemini_api_key)
# Using a semaphore to throttle the scribe phase
scribe_limiter = RateLimiter(rpm_limit=config.scripting_rpm)

class ScriptingAgent:
    def __init__(self, book_path: str):
        self.book_path = Path(book_path)
        self.output_dir = config.output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def get_model_for_task(self, task_type: str) -> str:
        """Select appropriate model based on task requirements."""
        model_map = {
            "global_context": config.scripting_model_global_context,
            "chapter_map": config.scripting_model_chapter_map,
            "page_script": config.scripting_model_page_script,
        }
        return model_map.get(task_type, config.scripting_model_page_script)

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

        print("🗺️  Generating Chapter Index with Beat Sheet...")
        prompt = """
        Act as a Master Storyboard Artist and Screenwriter. 
        Analyze the provided text and create a 'Beat Sheet' for a Graphic Novel.

        For each chapter, identify:
        1. 'chapter_number': Integer.
        2. 'title': Short descriptive title.
        3. 'start_phrase': The first 10-15 words of the chapter (for text slicing).
        4. 'end_phrase': The last 10-15 words of the chapter (for text slicing).
        5. 'estimated_word_count': Approximate word count.
        6. 'scene_type': 'action', 'dialogue', or 'contemplative'.
        7. 'narrative_arc': (Setup, Inciting Incident, Rising Action, Climax, or Resolution).
        8. 'beats': A list of emotional beats. Format: {"description": "...", "emotional_shift": "from X to Y"}.
        9. 'visual_theme': A recurring visual element for this chapter (e.g., 'claustrophobic interior', 'vast cold ocean').
        10. 'dialogue_distillation': The single most important line of dialogue that must be preserved.

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
                            "estimated_word_count": {"type": "INTEGER"},
                            "scene_type": {"type": "STRING", "enum": ["action", "dialogue", "contemplative"]},
                            "narrative_arc": {"type": "STRING"},
                            "beats": {
                                "type": "ARRAY",
                                "items": {
                                    "type": "OBJECT",
                                    "properties": {
                                        "description": {"type": "STRING"},
                                        "emotional_shift": {"type": "STRING"}
                                    }
                                }
                            },
                            "visual_theme": {"type": "STRING"},
                            "dialogue_distillation": {"type": "STRING"}
                        },
                        "required": ["chapter_number", "title", "start_phrase", "end_phrase", "scene_type", "narrative_arc", "beats", "visual_theme"]
                    }
                }
            )
        )
        
        chapter_map_path = self.output_dir / f"{self.book_path.stem}_chapter_map.json"
        with open(chapter_map_path, "w") as f:
            json.dump(response.parsed, f, indent=2)
            
        print(f"✅ Chapter Map Saved: {chapter_map_path}")
        return response.parsed

    def get_chapter_text(self, full_text: str, chapter: dict, next_chapter: dict = None) -> str:
        """
        Extract chapter text robustly using start/end phrases or next chapter start.
        """
        def find_fuzzy(text, phrase, start_pos=0):
            if not phrase:
                return -1
            words = phrase.strip().split()
            if not words:
                return -1
            # Create pattern: word followed by one or more whitespace characters
            pattern_str = r'\s+'.join(map(re.escape, words))
            match = re.search(pattern_str, text[start_pos:])
            if match:
                return match.start() + start_pos
            return -1

        # 1. Find Start
        start_phrase = chapter.get('start_phrase', '')
        start_idx = find_fuzzy(full_text, start_phrase)
        
        if start_idx == -1:
            # Fallback: simple string find
            start_idx = full_text.find(start_phrase)
            
        if start_idx == -1:
            print(f"⚠️ Could not locate start phrase for Ch {chapter['chapter_number']}: '{start_phrase[:30]}...'")
            return ""

        # 2. Find End
        end_idx = -1
        end_phrase = chapter.get('end_phrase', '')
        
        # Try specific end phrase first
        found_end_match_start = find_fuzzy(full_text, end_phrase, start_idx)
        if found_end_match_start != -1:
             # We want the END of the phrase
             # Re-match to get length
             match = re.search(r'\s+'.join(map(re.escape, end_phrase.split())), full_text[found_end_match_start:])
             if match:
                 end_idx = found_end_match_start + match.end()

        # 3. Fallback to Next Chapter Start if end phrase missing
        if end_idx == -1 and next_chapter:
            next_start_phrase = next_chapter.get('start_phrase', '')
            next_start_idx = find_fuzzy(full_text, next_start_phrase, start_idx)
            if next_start_idx != -1:
                end_idx = next_start_idx
                print(f"  ℹ️  Used next chapter start as boundary for Ch {chapter['chapter_number']}")

        # 4. Final Fallback (just take a chunk if nothing else works)
        if end_idx == -1:
            print(f"⚠️ Could not locate end boundary for Ch {chapter['chapter_number']}. taking 20k chars.")
            end_idx = min(len(full_text), start_idx + 20000)

        return full_text[start_idx:end_idx].strip()

    def allocate_pages_arc_aware(self, chapter_index: list, target_page_count: int) -> list:
        """
        Arc-aware page allocation using scene type and narrative position weights.
        Climax chapters get more pages; setup chapters get slightly fewer.

        Returns: List of page allocations with chapter assignments and PAGE GOALS.
        """
        total_chapters = len(chapter_index)

        # Calculate weighted word counts
        weighted_counts = []
        for ch in chapter_index:
            base_words = ch.get('estimated_word_count', 1000)
            scene_weight = ARC_WEIGHTS.get(ch.get('scene_type', 'dialogue'), 1.0)
            position_weight = get_position_weight(ch['chapter_number'], total_chapters)
            weighted_counts.append(base_words * scene_weight * position_weight)

        total_weighted = sum(weighted_counts)
        total_words = sum(ch.get('estimated_word_count', 1000) for ch in chapter_index)

        print(f"📊 Arc-Aware Page Allocation: {total_words:,} words across {total_chapters} chapters")
        print(f"   Using scene-type and three-act position weights")

        # Allocate pages proportionally to weighted counts
        page_allocations = []
        current_page = 1

        for i, chapter in enumerate(chapter_index):
            chapter_share = weighted_counts[i] / total_weighted
            chapter_pages = max(1, round(chapter_share * target_page_count))
            position_weight = get_position_weight(chapter['chapter_number'], total_chapters)
            
            arc = chapter.get('narrative_arc', 'Standard')

            for j in range(chapter_pages):
                if current_page > target_page_count:
                    break
                
                # Determine Page Goal based on position in chapter
                if j == 0:
                    page_goal = "Establish Scene & Atmosphere"
                elif j == chapter_pages - 1:
                    page_goal = "End on Cliffhanger/Hook"
                else:
                    page_goal = "Build Tension/Deepen Character"

                page_allocations.append({
                    'page_number': current_page,
                    'chapter_number': chapter['chapter_number'],
                    'chapter_title': chapter['title'],
                    'page_within_chapter': j + 1,
                    'total_pages_in_chapter': chapter_pages,
                    'arc_position': 'climax' if position_weight >= 1.3 else 'standard',
                    'narrative_arc': arc,
                    'page_goal': page_goal
                })
                current_page += 1

            if current_page > target_page_count:
                break

        # Trim to exact target (in case rounding gave us extra)
        page_allocations = page_allocations[:target_page_count]

        print(f"✅ Allocated {len(page_allocations)} pages across {total_chapters} chapters")
        return page_allocations


    @retry_with_backoff()
    async def write_page_script(
        self,
        page_allocation: dict,  # Simple page allocation with chapter info
        full_text: str,         # FULL book text
        chapter_index: list,    # Minimal structural index
        style: str,
        context_constraints: str = "",
        dialogue_mode: str = "balanced",
        previous_context: PageContext = None  # Rolling context from previous page
    ):
        """
        THE SCRIBE: Writes the detailed panel directions for a SINGLE page with FULL TEXT ACCESS.
        Now uses Beat-Based Adaptation.
        """
        async with scribe_limiter:
            # Get chapter info
            chapter_num = page_allocation['chapter_number']
            chapter = next(ch for ch in chapter_index if ch['chapter_number'] == chapter_num)
            
            # Find next chapter for boundary
            next_ch_struct = next((ch for ch in chapter_index if ch['chapter_number'] == chapter_num + 1), None)

            print(f"✍️  Scripting Page {page_allocation['page_number']}: Chapter {chapter_num} ({chapter['title']}) - Page {page_allocation['page_within_chapter']}/{page_allocation['total_pages_in_chapter']}...")

            # Extract full chapter text using robust method
            chapter_text = self.get_chapter_text(full_text, chapter, next_ch_struct)

            # Also include adjacent chapters for context (prevent boundary issues)
            prev_chapter_text = ""
            next_chapter_text = ""

            if chapter_num > 1:
                prev_ch = next((ch for ch in chapter_index if ch['chapter_number'] == chapter_num - 1), None)
                if prev_ch:
                    # We don't need next_ch for prev_ch extraction, just best effort
                    prev_full = self.get_chapter_text(full_text, prev_ch, chapter)
                    prev_chapter_text = prev_full # Pass full previous chapter

            if next_ch_struct:
                # We need next-next for next extraction? Just use None, it's context.
                next_full = self.get_chapter_text(full_text, next_ch_struct, None)
                next_chapter_text = next_full # Pass full next chapter

            # Build context
            full_context = f"""
PREVIOUS CHAPTER (for context):
{prev_chapter_text}

CURRENT CHAPTER TEXT:
{chapter_text}

NEXT CHAPTER (for context):
{next_chapter_text}
"""

            # Build continuity context from previous page
            continuity_context = ""
            if previous_context and previous_context.last_panel_summary:
                char_states_str = json.dumps(previous_context.character_states, indent=2) if previous_context.character_states else "No tracked characters yet."
                threads_str = "\n".join(f"- {t}" for t in previous_context.open_dialogue_threads) if previous_context.open_dialogue_threads else "None."
                continuity_context = f"""
CONTINUITY FROM PREVIOUS PAGE:
Last panel: {previous_context.last_panel_summary}

Character states entering this page:
{char_states_str}

Open dialogue/narrative threads:
{threads_str}

Scene momentum: {previous_context.scene_momentum}

CRITICAL: Maintain visual and narrative continuity with the above.
"""

            prompt = f"""
Act as an Award-Winning Graphic Novel Scriptwriter.

TASK:
Adapt Chapter {chapter_num} into Page {page_allocation['page_number']}.
You are not just copying text; you are translating prose into a visual medium.

STORYTELLING PRINCIPLES:
- SHOW, DON'T TELL: If the prose says "He was angry," describe a panel of a clenched fist or a shattered glass.
- COMPRESSION: One panel must often do the work of three paragraphs. 
- DIALOGUE: Distill long speeches into 'comic-sized' bubbles (max 25 words per bubble). Preserve the AUTHOR'S VOICE, but remove fluff.
- THE PAGE TURN: The final panel of this page MUST create curiosity or tension to make the reader turn the page.

CONTINUITY & ARC:
{continuity_context}
Chapter Arc: {page_allocation.get('narrative_arc', 'Standard')}
Page Goal: {page_allocation.get('page_goal', 'Move story forward')}
Visual Theme: {chapter.get('visual_theme', 'Standard')}

PANEL COMPOSITION GUIDANCE:
- Vary the "Camera Angle": (Wide Shot for setting, Close-up for emotion, Birds-eye for scale).
- Panel 1: Must establish the "Where" and "Who" if the scene changed.
- Last Panel: Must be a "Hook."
- ACTION scenes: Use 2-3 large dynamic panels
- DIALOGUE scenes: Use 4-6 panels for conversation flow

OUTPUT REQUIREMENTS:
- 'visual_description': Focus on cinematic lighting, character expression, and 'Mise-en-scène'.
- 'dialogue': Naturalistic, punchy speech. No 'Narrator' tags inside the string.
- 'caption': Use for internal monologue, narration, or time/location stamps.
- 'characters': List specifically who is in the panel.

CRITICAL: The FULL CHAPTER TEXT is provided below. Read it carefully and adapt dialogue
and scenes directly from it, distilling for impact.

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
                                    "panels": {
                                        "type": "ARRAY",
                                        "items": {
                                            "type": "OBJECT",
                                            "properties": {
                                                "panel_id": {"type": "INTEGER"},
                                                "visual_description": {"type": "STRING"},
                                                "dialogue": {"type": "STRING"},
                                                "caption": {"type": "STRING"},
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
                                                "bubble_position": {"type": "STRING", "enum": ["top-left", "top-right", "bottom-left", "bottom-right", "caption-box"]}
                                            },
                                            "required": ["panel_id", "visual_description", "advice", "characters", "bubble_position"]
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
                        print(f"⚠️ Model {model_name} rate limited on Page {page_allocation['page_number']}. Trying fallback...")
                        continue
                    raise e
            
            raise last_error

    async def generate_script(self, style: str, test_mode=True, context_constraints: str = "", target_page_override: int = None):
        """
        Main orchestration method: Generates complete script using simplified pipeline.
        """
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
        # Generate minimal chapter index for structural boundaries
        print("🗺️  Generating Chapter Index...")
        chapter_index = await self.generate_chapter_map(full_text)

        # 2. ARC-AWARE PAGE ALLOCATION
        # Allocate pages using scene type and narrative position weights
        print(f"📊 Allocating {target_pages} pages across chapters...")
        page_allocations = self.allocate_pages_arc_aware(chapter_index, target_pages)

        # 3. THE SCRIBE PHASE - Sequential with rolling context
        # Generate pages sequentially to maintain cross-page continuity
        print(f"✍️  Scripting {len(page_allocations)} pages sequentially with cross-page context...")

        full_script = []
        rolling_context = PageContext()

        for page_alloc in page_allocations:
            page = await self.write_page_script(
                page_alloc,
                full_text,
                chapter_index,
                style,
                context_constraints,
                dialogue_mode='balanced',
                previous_context=rolling_context
            )
            full_script.append(page)

            # Extract context for next page
            rolling_context = extract_context_from_page(page, rolling_context)

        # Validate dialogue fidelity (warning only, doesn't block)
        self._validate_dialogue_fidelity(full_script, chapter_index, full_text)

        # Save the result
        with open(output_file, "w") as f:
            json.dump(full_script, f, indent=2)

        print(f"✅ Script Generation Complete! Saved to: {output_file}")
        return full_script

    def _validate_dialogue_fidelity(self, script: list, chapter_index: list, full_text: str):
        """
        Checks if dialogue appears in source text.
        Relaxed for adaptation: simply logs stats rather than warning heavily about 'synthetic' dialogue.
        """
        print("\n🔍 Validating dialogue fidelity...")
        adapted_count = 0
        total_dialogue = 0

        for page in script:
            for panel in page.get('panels', []):
                dialogue = panel.get('dialogue', '')
                if not dialogue or len(dialogue.strip()) < 5:
                    continue
                
                total_dialogue += 1
                
                # Check if most of the dialogue words appear in sequence in the source
                dialogue_clean = dialogue.strip().replace('"', '').replace("'", "")
                words = dialogue_clean.split()
                
                if len(words) < 3:
                    continue

                search_phrase = ' '.join(words[:min(8, len(words))])
                if search_phrase.lower() not in full_text.lower():
                    adapted_count += 1

        print(f"  ℹ️  Dialogue Stats: {adapted_count}/{total_dialogue} lines appear to be adapted/distilled.")
        print("  (This is expected for a graphic novel adaptation).")

if __name__ == "__main__":
    agent = ScriptingAgent("assets/input/20-thousand-leagues-under-the-sea.txt")
    asyncio.run(agent.generate_script(style="Watercolor", test_mode=True))

