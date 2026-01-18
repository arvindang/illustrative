import os
import re
import json
import asyncio
import hashlib
from pathlib import Path
from dataclasses import dataclass, field
from dotenv import load_dotenv

# Load .env BEFORE importing config
load_dotenv()

from google.genai import types
from utils import (
    retry_with_backoff, RateLimiter, get_tpm_limiter,
    estimate_tokens_for_text, estimate_tokens_for_cache, extract_token_usage,
    retry_api_call, calculate_dynamic_timeout, calculate_beat_density,
    fix_era_anachronisms, get_client
)
from config import config

# Rate limiter for the parallel script writing phase (lazy-initialized)
# Uses effective rate limits based on Vertex AI vs AI Studio
_scribe_limiter = None

def get_scribe_limiter() -> RateLimiter:
    """Get the scripting rate limiter with effective RPM based on config."""
    global _scribe_limiter
    if _scribe_limiter is None:
        effective_limits = config.get_effective_rate_limits()
        _scribe_limiter = RateLimiter(rpm_limit=effective_limits['scripting_rpm'])
    return _scribe_limiter

class ScriptingAgent:
    def __init__(self, book_path: str, base_output_dir: Path = None):
        self.book_path = Path(book_path)
        self.output_dir = Path(base_output_dir) if base_output_dir else config.output_dir
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

        # Acquire TPM capacity for cache creation
        estimated_tokens = estimate_tokens_for_cache(content)
        await get_tpm_limiter().acquire(estimated_tokens)

        try:
            # Note: The SDK for caching might vary slightly by version. 
            # We are using the 'google.genai' library.
            
            # 1. Create the cache
            # We use the 'scripting_model_global_context' as the base model for the cache
            # or a known caching-capable model like gemini-1.5-pro-002
            
            # Using a distinct display name helps debugging
            display_name = f"book_cache_{self.book_path.stem}_{content_hash[:6]}"
            
            cached_content = await get_client().aio.caches.create(
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

    def normalize_character_name(self, name: str, registry: dict):
        """
        Normalizes character names to handle variants and duplicates.
        Returns canonical name and folder-safe name.
        """
        if not name:
            return None, None

        titles = ["captain", "professor", "doctor", "dr", "mr", "mrs", "miss", "sir", "lord", "lady"]
        parts = name.strip().split()
        base_name = parts[-1].lower() if parts else ""

        # Check if this base name already has a canonical form
        for canonical, variants in registry.items():
            canonical_base = canonical.split()[-1].lower()
            if base_name == canonical_base:
                registry[canonical].add(name)
                folder_name = canonical.lower().replace(" ", "_")
                return canonical, folder_name

        # New character - prefer names with titles
        has_title = any(part.lower() in titles for part in parts[:-1])
        canonical = name if has_title or len(parts) > 1 else name
        registry[canonical] = {name}
        folder_name = canonical.lower().replace(" ", "_")
        return canonical, folder_name

    def extract_characters_from_blueprint(self, blueprint: list):
        """Extracts unique, normalized character names from the blueprint."""
        registry = {}
        canonical_chars = []

        for page in blueprint:
            for char_name in page.get('key_characters', []):
                canonical, _ = self.normalize_character_name(char_name, registry)
                if canonical and canonical not in canonical_chars:
                    canonical_chars.append(canonical)

        return canonical_chars

    @retry_with_backoff()
    async def analyze_narrative_beats(self, cache_name: str, full_text_fallback: str, target_pages: int) -> dict:
        """
        PASS 1: BEAT ANALYSIS (Enhanced)
        Analyzes the narrative structure and identifies story beats with visual potential scoring.

        This pass breaks the novel into narrative beats with intensity scores,
        identifies act structure, calculates page allocation, and scores visual potential.

        Key enhancements:
        - visual_potential: How well each beat translates to visual storytelling
        - adaptation_notes: What works visually vs textually
        - suggested_focus: Specific visual moments to highlight
        - micro_beats: Panel-level pacing suggestions within each beat

        Args:
            cache_name: Gemini context cache reference
            full_text_fallback: Full text if cache unavailable
            target_pages: Target number of pages for allocation

        Returns:
            BeatMap dict with beats, act_boundaries, pacing_recommendations, hooks
        """
        print(f"🎭 BEAT ANALYSIS PASS: Extracting narrative structure with visual potential...")

        prompt = f"""
        Act as a Story Structure Analyst AND Visual Storytelling Expert for graphic novel adaptation.

        TASK:
        Analyze the source material and identify the key narrative BEATS.
        A beat is a significant story event that advances the plot or reveals character.

        For a {target_pages}-page graphic novel, identify approximately {max(8, target_pages // 4)} major beats.
        (More granular than traditional story beats - each beat should map to 3-5 pages)

        For EACH beat, provide:
        1. 'beat_id': Sequential integer starting from 1
        2. 'beat_type': One of: "inciting", "rising", "midpoint", "crisis", "climax", "resolution", "denouement", "transition"
        3. 'description': A 1-2 sentence description of what happens in this beat
        4. 'intensity': Float from 0.0 to 1.0 indicating dramatic intensity
           - 0.0-0.3: Quiet moments, transitions, establishing shots
           - 0.4-0.6: Rising action, character development
           - 0.7-0.8: Major confrontations, revelations
           - 0.9-1.0: Climax, crisis, major turning points
        5. 'key_characters': List of characters central to this beat
        6. 'emotional_tone': The dominant emotion (e.g., "tense", "melancholic", "triumphant")
        7. 'scene_type': One of: "action", "dialogue", "establishing", "montage", "flashback", "transition"

        NEW - VISUAL ADAPTATION FIELDS:
        8. 'visual_potential': Float from 0.0 to 1.0 indicating how well this beat translates to visual medium
           - 0.0-0.3: Internal/abstract (thoughts, philosophy, internal monologue) - HARD to visualize
           - 0.4-0.6: Mixed (conversation with some action) - MODERATE visual potential
           - 0.7-0.8: Visual events (discoveries, arrivals, confrontations) - GOOD visual potential
           - 0.9-1.0: Highly visual (action sequences, reveals, transformations) - EXCELLENT visual potential
        9. 'adaptation_notes': Brief note on what to emphasize visually vs what to condense/cut
        10. 'suggested_focus': List of 2-3 specific VISUAL MOMENTS worth highlighting as panels
           (e.g., "Captain's face when he sees the creature", "The submarine emerging from darkness")
        11. 'micro_beats': List of 3-6 panel-level micro-events within this beat
           (Each micro_beat is a single panel-worthy moment)
        12. 'is_page_turn_hook': Boolean - true if this beat should end on a page turn for suspense

        Also identify:
        - 'act_boundaries': Which beat_id ends Act 1 and Act 2 (for 3-act structure)
        - 'total_word_count': Approximate word count of source material
        - 'hooks': List of beat_ids that are compelling "return" moments (cliffhangers, reveals)
        - 'low_visual_warnings': List of beat_ids with visual_potential < 0.4 that may need creative adaptation

        OUTPUT: JSON object with the specified structure.
        """

        model = config.scripting_model_global_context

        if cache_name:
            contents = [prompt]
            cached_content = cache_name
        else:
            contents = [prompt, f"SOURCE BOOK:\n{full_text_fallback}"]
            cached_content = None

        # Acquire TPM capacity
        estimated_tokens = estimate_tokens_for_text(prompt)
        await get_tpm_limiter().acquire(estimated_tokens)

        response = await get_client().aio.models.generate_content(
            model=model,
            contents=contents,
            config=types.GenerateContentConfig(
                cached_content=cached_content,
                response_mime_type="application/json",
                response_schema={
                    "type": "OBJECT",
                    "properties": {
                        "total_word_count": {"type": "INTEGER"},
                        "beats": {
                            "type": "ARRAY",
                            "items": {
                                "type": "OBJECT",
                                "properties": {
                                    "beat_id": {"type": "INTEGER"},
                                    "beat_type": {"type": "STRING"},
                                    "description": {"type": "STRING"},
                                    "intensity": {"type": "NUMBER"},
                                    "key_characters": {"type": "ARRAY", "items": {"type": "STRING"}},
                                    "emotional_tone": {"type": "STRING"},
                                    "scene_type": {"type": "STRING"},
                                    # New visual adaptation fields
                                    "visual_potential": {"type": "NUMBER"},
                                    "adaptation_notes": {"type": "STRING"},
                                    "suggested_focus": {"type": "ARRAY", "items": {"type": "STRING"}},
                                    "micro_beats": {"type": "ARRAY", "items": {"type": "STRING"}},
                                    "is_page_turn_hook": {"type": "BOOLEAN"}
                                },
                                "required": ["beat_id", "beat_type", "description", "intensity", "key_characters", "visual_potential"]
                            }
                        },
                        "act_boundaries": {
                            "type": "OBJECT",
                            "properties": {
                                "act_1_end": {"type": "INTEGER"},
                                "act_2_end": {"type": "INTEGER"}
                            }
                        },
                        "hooks": {"type": "ARRAY", "items": {"type": "INTEGER"}},
                        "low_visual_warnings": {"type": "ARRAY", "items": {"type": "INTEGER"}}
                    },
                    "required": ["beats", "act_boundaries"]
                }
            )
        )

        # Update TPM with actual usage
        input_tokens, output_tokens = extract_token_usage(response)
        get_tpm_limiter().update_actual_usage(estimated_tokens, input_tokens + output_tokens)

        if response.parsed is None:
            raise ValueError("Failed to parse beat analysis from API response")

        beat_map = response.parsed

        # Calculate page allocation from beat density
        beats = beat_map.get('beats', [])
        pages_per_beat = calculate_beat_density(beats, target_pages)

        # Add pacing recommendations to the beat map
        beat_map['pacing_recommendations'] = {
            'pages_per_beat': pages_per_beat,
            'target_pages': target_pages
        }

        # Add page_allocation to each beat for convenience
        for beat in beats:
            beat_id = beat.get('beat_id', 0)
            beat['page_allocation'] = pages_per_beat.get(beat_id, 1)

        # Log visual adaptation insights
        hooks = beat_map.get('hooks', [])
        low_visual = beat_map.get('low_visual_warnings', [])
        high_visual_beats = [b for b in beats if b.get('visual_potential', 0) >= 0.7]

        print(f"✅ Identified {len(beats)} narrative beats")
        print(f"   Act 1 ends at beat {beat_map.get('act_boundaries', {}).get('act_1_end', '?')}")
        print(f"   Act 2 ends at beat {beat_map.get('act_boundaries', {}).get('act_2_end', '?')}")
        print(f"   📷 High visual potential beats: {len(high_visual_beats)}/{len(beats)}")
        if hooks:
            print(f"   🎣 Page-turn hooks identified: beats {hooks}")
        if low_visual:
            print(f"   ⚠️  Low visual potential warnings: beats {low_visual} (may need creative adaptation)")

        return beat_map

    @retry_with_backoff()
    async def generate_adaptation_filter(self, cache_name: str, full_text_fallback: str, beat_map: dict, target_pages: int) -> dict:
        """
        PASS 1.5: ADAPTATION FILTER (NEW)
        Identifies what content to cut, condense, or keep for the graphic novel adaptation.

        This pass acts as an editorial filter, making creative decisions about:
        - What scenes are ESSENTIAL and must be kept
        - What scenes can be CONDENSED (merged or summarized visually)
        - What scenes should be CUT entirely (internal monologue, repetitive content)
        - What needs CREATIVE ADAPTATION (internal thoughts shown through expression/action)

        Key insight: Not all prose translates well to visual storytelling. This pass
        identifies natural "breaking points" and preserves what makes readers love the story.

        Args:
            cache_name: Gemini context cache reference
            full_text_fallback: Full text if cache unavailable
            beat_map: Beat analysis from Pass 1
            target_pages: Target number of pages

        Returns:
            AdaptationFilter dict with scene classifications and recommendations
        """
        print(f"✂️  ADAPTATION FILTER: Identifying what to keep, condense, and cut...")

        # Build beat summary for context
        beats = beat_map.get('beats', [])
        beat_summary = "\n".join([
            f"  Beat {b['beat_id']}: {b['description']} (visual: {b.get('visual_potential', 0.5):.1f})"
            for b in beats[:20]  # Limit to first 20 beats for prompt size
        ])

        prompt = f"""
        Act as a Professional Graphic Novel Adapter and Editor.

        You are adapting a novel into a {target_pages}-page graphic novel. Your job is to make
        editorial decisions about what to KEEP, CONDENSE, or CUT.

        BEAT ANALYSIS (from prior pass):
{beat_summary}

        TASK:
        Analyze the source material and classify each major scene/sequence into:

        1. 'essential_scenes': Scenes that MUST be kept (pivotal plot points, character-defining moments)
           - For each: provide 'description', 'beat_ids' it covers, 'why_essential'

        2. 'condensable_scenes': Scenes that can be merged or summarized
           - For each: provide 'description', 'beat_ids', 'condensation_strategy'
           - Strategies: "montage" (show time passing), "single_panel" (summarize in one image),
             "merge_with" (combine with adjacent scene), "dialogue_summary" (one caption covers it)

        3. 'cuttable_scenes': Scenes that should be CUT entirely
           - For each: provide 'description', 'beat_ids', 'reason_to_cut'
           - Reasons: "pure_internal" (unvisualizable thoughts), "redundant" (repeats earlier info),
             "tangential" (doesn't advance main plot), "better_told_elsewhere" (info covered in other scenes)

        4. 'creative_adaptations': Scenes that need creative visual translation
           - For each: provide 'description', 'beat_ids', 'adaptation_strategy'
           - Strategies: "show_through_expression" (internal state via facial/body language),
             "visual_metaphor" (abstract concept as image), "flashback_panel" (brief memory image),
             "environmental_storytelling" (show through setting details)

        5. 'reader_beloved_moments': The 3-5 moments that fans would be MOST disappointed to miss
           - These are non-negotiable MUST-INCLUDE moments regardless of visual potential

        6. 'pacing_recommendations':
           - 'slow_down': List of beat_ids that deserve more visual time (splash pages, etc.)
           - 'speed_up': List of beat_ids that should move quickly (action montages, transitions)

        7. 'natural_chapter_breaks': Suggested page numbers for natural "chapter" divisions
           (Where a reader could put the book down and feel satisfied)

        OUTPUT: JSON object with the specified structure.
        Be ruthless but thoughtful - a great adaptation knows what to leave out.
        """

        model = config.scripting_model_global_context

        if cache_name:
            contents = [prompt]
            cached_content = cache_name
        else:
            contents = [prompt, f"SOURCE BOOK:\n{full_text_fallback[:80000]}"]
            cached_content = None

        # Acquire TPM capacity
        estimated_tokens = estimate_tokens_for_text(prompt)
        await get_tpm_limiter().acquire(estimated_tokens)

        response = await get_client().aio.models.generate_content(
            model=model,
            contents=contents,
            config=types.GenerateContentConfig(
                cached_content=cached_content,
                response_mime_type="application/json",
                response_schema={
                    "type": "OBJECT",
                    "properties": {
                        "essential_scenes": {
                            "type": "ARRAY",
                            "items": {
                                "type": "OBJECT",
                                "properties": {
                                    "description": {"type": "STRING"},
                                    "beat_ids": {"type": "ARRAY", "items": {"type": "INTEGER"}},
                                    "why_essential": {"type": "STRING"}
                                },
                                "required": ["description", "beat_ids", "why_essential"]
                            }
                        },
                        "condensable_scenes": {
                            "type": "ARRAY",
                            "items": {
                                "type": "OBJECT",
                                "properties": {
                                    "description": {"type": "STRING"},
                                    "beat_ids": {"type": "ARRAY", "items": {"type": "INTEGER"}},
                                    "condensation_strategy": {"type": "STRING"}
                                },
                                "required": ["description", "beat_ids", "condensation_strategy"]
                            }
                        },
                        "cuttable_scenes": {
                            "type": "ARRAY",
                            "items": {
                                "type": "OBJECT",
                                "properties": {
                                    "description": {"type": "STRING"},
                                    "beat_ids": {"type": "ARRAY", "items": {"type": "INTEGER"}},
                                    "reason_to_cut": {"type": "STRING"}
                                },
                                "required": ["description", "beat_ids", "reason_to_cut"]
                            }
                        },
                        "creative_adaptations": {
                            "type": "ARRAY",
                            "items": {
                                "type": "OBJECT",
                                "properties": {
                                    "description": {"type": "STRING"},
                                    "beat_ids": {"type": "ARRAY", "items": {"type": "INTEGER"}},
                                    "adaptation_strategy": {"type": "STRING"}
                                },
                                "required": ["description", "beat_ids", "adaptation_strategy"]
                            }
                        },
                        "reader_beloved_moments": {
                            "type": "ARRAY",
                            "items": {
                                "type": "OBJECT",
                                "properties": {
                                    "description": {"type": "STRING"},
                                    "beat_ids": {"type": "ARRAY", "items": {"type": "INTEGER"}},
                                    "why_beloved": {"type": "STRING"}
                                },
                                "required": ["description", "why_beloved"]
                            }
                        },
                        "pacing_recommendations": {
                            "type": "OBJECT",
                            "properties": {
                                "slow_down": {"type": "ARRAY", "items": {"type": "INTEGER"}},
                                "speed_up": {"type": "ARRAY", "items": {"type": "INTEGER"}}
                            }
                        },
                        "natural_chapter_breaks": {"type": "ARRAY", "items": {"type": "INTEGER"}}
                    },
                    "required": ["essential_scenes", "condensable_scenes", "cuttable_scenes", "reader_beloved_moments"]
                }
            )
        )

        # Update TPM with actual usage
        input_tokens, output_tokens = extract_token_usage(response)
        get_tpm_limiter().update_actual_usage(estimated_tokens, input_tokens + output_tokens)

        if response.parsed is None:
            raise ValueError("Failed to parse adaptation filter from API response")

        adaptation_filter = response.parsed

        # Log summary
        essential_count = len(adaptation_filter.get('essential_scenes', []))
        condensable_count = len(adaptation_filter.get('condensable_scenes', []))
        cuttable_count = len(adaptation_filter.get('cuttable_scenes', []))
        beloved_count = len(adaptation_filter.get('reader_beloved_moments', []))

        print(f"✅ Adaptation filter complete:")
        print(f"   ✓ Essential scenes: {essential_count}")
        print(f"   ↔ Condensable scenes: {condensable_count}")
        print(f"   ✗ Cuttable scenes: {cuttable_count}")
        print(f"   ❤️  Reader-beloved moments: {beloved_count}")

        pacing = adaptation_filter.get('pacing_recommendations', {})
        if pacing.get('slow_down'):
            print(f"   🐢 Slow down for beats: {pacing['slow_down']}")
        if pacing.get('speed_up'):
            print(f"   🏃 Speed up through beats: {pacing['speed_up']}")

        return adaptation_filter

    @retry_with_backoff()
    async def generate_character_deep_dive(self, cache_name: str, full_text_fallback: str, blueprint: list) -> dict:
        """
        PASS 3: CHARACTER DEEP DIVE
        Generates detailed character arcs and scene-specific states.

        Args:
            cache_name: Gemini context cache reference
            full_text_fallback: Full text if cache unavailable
            blueprint: Page-by-page blueprint from Pass 2

        Returns:
            CharacterArcs dict with characters, scene_states
        """
        # Extract unique characters from blueprint
        characters = self.extract_characters_from_blueprint(blueprint)
        print(f"🎭 CHARACTER DEEP DIVE: Analyzing arcs for {len(characters)} characters...")

        # Create a summary of pages for context
        page_summaries = [
            f"Page {p['page_number']}: {p.get('summary', '')} (Characters: {', '.join(p.get('key_characters', []))})"
            for p in blueprint[:50]  # Limit to first 50 pages for context size
        ]
        page_context = "\n".join(page_summaries)

        prompt = f"""
        Act as a Character Development Specialist for graphic novel adaptation.

        TASK:
        Analyze the characters and their journeys through the story.

        CHARACTERS TO ANALYZE:
        {', '.join(characters)}

        PAGE SUMMARY (for tracking character appearances):
        {page_context}

        For EACH major character (top 5-8), provide:
        1. 'name': Canonical name
        2. 'role': "protagonist", "antagonist", "supporting", or "minor"
        3. 'arc_type': "transformation", "flat", "corruption", "redemption", "fall", "rise"
        4. 'introduction_page': First page number where they appear
        5. 'distinctive_items': List of 2-3 items they should always carry/wear
        6. 'era_appropriate_gear': Object mapping scene types to required gear:
           - "underwater": ["diving helmet", "weighted boots", etc.]
           - "aboard_ship": ["nautical uniform", etc.]
           - "formal": ["evening wear", etc.]
           - "action": ["practical clothing", etc.]
        7. 'relationships': Object mapping other character names to relationship types
           (e.g., "Captain Nemo": "mentor-student", "Ned Land": "reluctant-ally")
        8. 'key_moments': List of {{page, event, emotional_state, visual_change}} for major character beats

        NEW - VOICE & DIALECT TRACKING (for dialogue consistency):
        9. 'voice_profile': Object with:
           - 'education_level': "erudite", "educated", "working_class", "uneducated"
           - 'formality': "very_formal", "formal", "casual", "rough"
           - 'vocabulary_style': Brief description (e.g., "scientific terminology", "nautical jargon", "simple direct")
           - 'dialect_markers': List of speech patterns (e.g., "drops articles", "French accent shown via occasional French words", "uses 'ain't'")
           - 'emotional_tells': How their speech changes under stress/excitement
           - 'catchphrases': List of phrases they repeat (max 3)
        10. 'dialogue_samples': List of 3-5 example dialogue lines that capture their voice
            (These will be used as reference when writing their dialogue)
        11. 'speech_contrast': How this character's speech differs from other main characters
            (e.g., "More verbose than Ned Land", "Uses technical terms unlike other characters")

        Also provide 'scene_states' - a list of page-by-page character states for key pages:
        For each important scene transition (every 5-10 pages or major scene change):
        - 'page_number': The page number
        - 'characters': Object mapping character names to their state:
          - 'emotional_state': Current emotion
          - 'gear': List of items they should have in this scene
          - 'notes': Any special visual requirements
        - 'interaction_rules': List of rules for how characters interact in this scene

        OUTPUT: JSON object with 'characters' array and 'scene_states' array.
        """

        model = config.scripting_model_global_context

        if cache_name:
            contents = [prompt]
            cached_content = cache_name
        else:
            contents = [prompt, f"SOURCE BOOK:\n{full_text_fallback[:50000]}"]
            cached_content = None

        # Acquire TPM capacity
        estimated_tokens = estimate_tokens_for_text(prompt)
        await get_tpm_limiter().acquire(estimated_tokens)

        response = await get_client().aio.models.generate_content(
            model=model,
            contents=contents,
            config=types.GenerateContentConfig(
                cached_content=cached_content,
                response_mime_type="application/json",
                response_schema={
                    "type": "OBJECT",
                    "properties": {
                        "characters": {
                            "type": "ARRAY",
                            "items": {
                                "type": "OBJECT",
                                "properties": {
                                    "name": {"type": "STRING"},
                                    "role": {"type": "STRING"},
                                    "arc_type": {"type": "STRING"},
                                    "introduction_page": {"type": "INTEGER"},
                                    "distinctive_items": {"type": "ARRAY", "items": {"type": "STRING"}},
                                    "era_appropriate_gear": {
                                        "type": "OBJECT",
                                        "additionalProperties": {
                                            "type": "ARRAY",
                                            "items": {"type": "STRING"}
                                        }
                                    },
                                    "relationships": {
                                        "type": "OBJECT",
                                        "additionalProperties": {"type": "STRING"}
                                    },
                                    "key_moments": {
                                        "type": "ARRAY",
                                        "items": {
                                            "type": "OBJECT",
                                            "properties": {
                                                "page": {"type": "INTEGER"},
                                                "event": {"type": "STRING"},
                                                "emotional_state": {"type": "STRING"},
                                                "visual_change": {"type": "STRING"}
                                            }
                                        }
                                    },
                                    # New voice/dialect tracking fields
                                    "voice_profile": {
                                        "type": "OBJECT",
                                        "properties": {
                                            "education_level": {"type": "STRING"},
                                            "formality": {"type": "STRING"},
                                            "vocabulary_style": {"type": "STRING"},
                                            "dialect_markers": {"type": "ARRAY", "items": {"type": "STRING"}},
                                            "emotional_tells": {"type": "STRING"},
                                            "catchphrases": {"type": "ARRAY", "items": {"type": "STRING"}}
                                        }
                                    },
                                    "dialogue_samples": {"type": "ARRAY", "items": {"type": "STRING"}},
                                    "speech_contrast": {"type": "STRING"}
                                },
                                "required": ["name", "role", "distinctive_items"]
                            }
                        },
                        "scene_states": {
                            "type": "ARRAY",
                            "items": {
                                "type": "OBJECT",
                                "properties": {
                                    "page_number": {"type": "INTEGER"},
                                    "characters": {
                                        "type": "OBJECT",
                                        "additionalProperties": {
                                            "type": "OBJECT",
                                            "properties": {
                                                "emotional_state": {"type": "STRING"},
                                                "gear": {"type": "ARRAY", "items": {"type": "STRING"}},
                                                "notes": {"type": "STRING"}
                                            }
                                        }
                                    },
                                    "interaction_rules": {"type": "ARRAY", "items": {"type": "STRING"}}
                                },
                                "required": ["page_number", "characters"]
                            }
                        }
                    },
                    "required": ["characters", "scene_states"]
                }
            )
        )

        # Update TPM with actual usage
        input_tokens, output_tokens = extract_token_usage(response)
        get_tpm_limiter().update_actual_usage(estimated_tokens, input_tokens + output_tokens)

        if response.parsed is None:
            raise ValueError("Failed to parse character deep dive from API response")

        character_arcs = response.parsed

        print(f"✅ Character arcs complete: {len(character_arcs.get('characters', []))} characters analyzed")
        print(f"   Scene states tracked: {len(character_arcs.get('scene_states', []))} key scenes")

        return character_arcs

    @retry_with_backoff()
    async def generate_asset_manifest(self, cache_name: str, full_text_fallback: str, blueprint: list, style: str, context_constraints: str = "", character_arcs: dict = None):
        """
        PASS 4: ASSET EXTRACTION (Enhanced)
        Generates visual descriptions for characters and key objects in a single pass.
        Now includes interaction_rules for scene-specific requirements.
        """
        # Extract unique characters from blueprint
        characters = self.extract_characters_from_blueprint(blueprint)
        print(f"👥 Found {len(characters)} unique characters: {', '.join(characters)}")

        # Build era/context constraint block if provided
        era_block = ""
        if context_constraints:
            era_block = f"""
        HISTORICAL/SETTING CONSTRAINTS (CRITICAL - ALL DESIGNS MUST CONFORM):
        {context_constraints}

        ALL character clothing, accessories, and object designs MUST be era-appropriate.
        NO anachronistic elements (modern clothing, technology, or materials).
        """

        # Build character arc context if available
        arc_context = ""
        if character_arcs and character_arcs.get('characters'):
            arc_list = []
            for char in character_arcs['characters'][:5]:  # Top 5 characters
                items = ', '.join(char.get('distinctive_items', []))
                arc_list.append(f"  - {char.get('name')}: {char.get('role', 'supporting')}, items: {items}")
            arc_context = f"""
        CHARACTER ARC CONTEXT (from deep dive analysis):
{chr(10).join(arc_list)}

        Ensure character designs incorporate these distinctive items and role characteristics.
        """

        # Combined prompt for characters, objects, locations, and color palette
        combined_prompt = f"""
        Act as a Visual Development Artist for a '{style}' graphic novel.
        {era_block}
        {arc_context}
        PART 1: CHARACTER DESIGN
        For each of the following characters, provide:
        - name: Canonical name
        - description: Detailed physical description (facial features, clothing, color palette). ALL clothing and accessories must be era-appropriate.
        - age_range: e.g., "mid-30s", "elderly", "young adult"
        - occupation: e.g., "Sea Captain", "Professor"
        - distinctive_items: List of items they carry or wear (must be era-appropriate)
        - specific_era_markers: Specific historical fashion details (e.g., "Victorian high collar", "19th-century heavy wool coat")
        - color_signature: 2-3 colors that define this character's visual identity (for consistency)

        Characters to design: {', '.join(characters)}

        PART 2: KEY OBJECTS
        Identify the top 3-5 most important RECURRING OBJECTS or VEHICLES that need consistent visual design.
        For each, provide:
        - name: Name of the object
        - description: Visual description (materials, textures, colors). Must be era-appropriate.
        - key_features: List of identifying shapes or mechanisms (era-appropriate technology)
        - condition: The state of wear (e.g., "Pristine and polished", "Rusted and barnacle-encrusted")
        - material_context: Primary materials appropriate to the era

        PART 3: RECURRING LOCATIONS (NEW)
        Identify the top 5-8 RECURRING LOCATIONS/ENVIRONMENTS that appear multiple times and need consistent visual design.
        For each location, provide:
        - name: Name of the location (e.g., "The Nautilus Bridge", "Professor's Study", "Underwater Coral Gardens")
        - description: Detailed visual description - architecture, lighting, atmosphere
        - color_palette: List of 3-4 dominant colors for this location
        - lighting: Typical lighting conditions (e.g., "Dim gas lamps with golden glow", "Bioluminescent blue-green")
        - recurring_elements: List of props/details that should ALWAYS appear in this location
        - mood: The emotional atmosphere (e.g., "Mysterious and claustrophobic", "Grand and awe-inspiring")
        - era_markers: Period-specific architectural/design details

        PART 4: INTERACTION RULES
        Define scene-specific requirements for visual consistency:
        - underwater_scenes: List requirements (e.g., "All characters MUST wear period diving equipment with brass helmets")
        - formal_scenes: List requirements (e.g., "Victorian evening wear, men in frock coats")
        - action_scenes: List requirements (e.g., "Practical clothing, freedom of movement")
        - aboard_ship: List requirements (e.g., "Nautical attire appropriate to role")

        Also identify any 'forbidden_combinations' - things that should NEVER appear together:
        - Example: {{"characters": ["Professor Aronnax"], "items": ["modern SCUBA gear"], "reason": "anachronism"}}

        PART 5: GLOBAL COLOR SCRIPT
        Define the overall color direction for the graphic novel:
        - primary_palette: List of 4-5 dominant colors for the entire work
        - act_1_colors: Colors that dominate the beginning (before the adventure intensifies)
        - act_2_colors: Colors for the middle (during the main conflict)
        - act_3_colors: Colors for the climax and resolution
        - color_associations: Object mapping emotions to colors (e.g., {{"danger": "deep red", "wonder": "teal blue"}})

        Be specific and visual for use in AI image generation.
        """

        model = config.scripting_model_global_context
        contents = [combined_prompt] if cache_name else [combined_prompt, f"SOURCE BOOK:\n{full_text_fallback[:50000]}"]

        # Acquire TPM capacity for combined manifest
        estimated_tokens = estimate_tokens_for_text(combined_prompt)
        await get_tpm_limiter().acquire(estimated_tokens)

        response = await get_client().aio.models.generate_content(
            model=model,
            contents=contents,
            config=types.GenerateContentConfig(
                cached_content=cache_name,
                response_mime_type="application/json",
                response_schema={
                    "type": "OBJECT",
                    "properties": {
                        "characters": {
                            "type": "ARRAY",
                            "items": {
                                "type": "OBJECT",
                                "properties": {
                                    "name": {"type": "STRING"},
                                    "description": {"type": "STRING"},
                                    "age_range": {"type": "STRING"},
                                    "occupation": {"type": "STRING"},
                                    "distinctive_items": {"type": "ARRAY", "items": {"type": "STRING"}},
                                    "specific_era_markers": {"type": "STRING"},
                                    "color_signature": {"type": "ARRAY", "items": {"type": "STRING"}}
                                },
                                "required": ["name", "description"]
                            }
                        },
                        "objects": {
                            "type": "ARRAY",
                            "items": {
                                "type": "OBJECT",
                                "properties": {
                                    "name": {"type": "STRING"},
                                    "description": {"type": "STRING"},
                                    "key_features": {"type": "ARRAY", "items": {"type": "STRING"}},
                                    "condition": {"type": "STRING"},
                                    "material_context": {"type": "STRING"}
                                },
                                "required": ["name", "description"]
                            }
                        },
                        "locations": {
                            "type": "ARRAY",
                            "items": {
                                "type": "OBJECT",
                                "properties": {
                                    "name": {"type": "STRING"},
                                    "description": {"type": "STRING"},
                                    "color_palette": {"type": "ARRAY", "items": {"type": "STRING"}},
                                    "lighting": {"type": "STRING"},
                                    "recurring_elements": {"type": "ARRAY", "items": {"type": "STRING"}},
                                    "mood": {"type": "STRING"},
                                    "era_markers": {"type": "STRING"}
                                },
                                "required": ["name", "description", "lighting", "mood"]
                            }
                        },
                        "interaction_rules": {
                            "type": "OBJECT",
                            "properties": {
                                "underwater_scenes": {"type": "ARRAY", "items": {"type": "STRING"}},
                                "formal_scenes": {"type": "ARRAY", "items": {"type": "STRING"}},
                                "action_scenes": {"type": "ARRAY", "items": {"type": "STRING"}},
                                "aboard_ship": {"type": "ARRAY", "items": {"type": "STRING"}}
                            }
                        },
                        "forbidden_combinations": {
                            "type": "ARRAY",
                            "items": {
                                "type": "OBJECT",
                                "properties": {
                                    "characters": {"type": "ARRAY", "items": {"type": "STRING"}},
                                    "items": {"type": "ARRAY", "items": {"type": "STRING"}},
                                    "reason": {"type": "STRING"}
                                }
                            }
                        },
                        "color_script": {
                            "type": "OBJECT",
                            "properties": {
                                "primary_palette": {"type": "ARRAY", "items": {"type": "STRING"}},
                                "act_1_colors": {"type": "ARRAY", "items": {"type": "STRING"}},
                                "act_2_colors": {"type": "ARRAY", "items": {"type": "STRING"}},
                                "act_3_colors": {"type": "ARRAY", "items": {"type": "STRING"}},
                                "color_associations": {
                                    "type": "OBJECT",
                                    "additionalProperties": {"type": "STRING"}
                                }
                            }
                        }
                    },
                    "required": ["characters", "objects", "locations", "interaction_rules"]
                }
            )
        )

        # Update TPM with actual usage
        input_tokens, output_tokens = extract_token_usage(response)
        get_tpm_limiter().update_actual_usage(estimated_tokens, input_tokens + output_tokens)

        if response.parsed is None:
            raw_text = response.text[:500] if response.text else 'empty'
            print(f"⚠️ Asset manifest response failed to parse. Raw text: {raw_text}")
            raise ValueError("Failed to parse combined asset manifest from API response")

        manifest = response.parsed
        print(f"🛠️ Asset manifest complete:")
        print(f"   👥 Characters: {len(manifest.get('characters', []))}")
        print(f"   📦 Objects: {len(manifest.get('objects', []))}")
        print(f"   🏠 Locations: {len(manifest.get('locations', []))}")
        if manifest.get('color_script', {}).get('primary_palette'):
            print(f"   🎨 Color palette: {', '.join(manifest['color_script']['primary_palette'][:4])}")
        return manifest

    # Threshold for chunked generation (pages above this get split into chunks)
    CHUNK_THRESHOLD = 40
    CHUNK_SIZE = 30  # Pages per chunk for large jobs

    async def _generate_blueprint_chunk(
        self,
        cache_name: str,
        full_text_fallback: str,
        start_page: int,
        end_page: int,
        total_pages: int,
        style: str,
        context_constraints: str = "",
        previous_summary: str = "",
        beat_guidance: str = "",
        adaptation_guidance: str = ""
    ):
        """
        Generate a chunk of the blueprint (internal helper).

        Args:
            cache_name: Gemini context cache name
            full_text_fallback: Full text if cache unavailable
            start_page: First page number in this chunk
            end_page: Last page number in this chunk
            total_pages: Total pages in the full novel
            style: Art style
            context_constraints: Era/setting constraints
            previous_summary: Summary of what happened in previous chunks (for continuity)
            beat_guidance: Beat-based pacing guidance
            adaptation_guidance: Adaptation filter guidance (what to keep/cut)
        """
        chunk_size = end_page - start_page + 1
        is_first_chunk = start_page == 1
        is_last_chunk = end_page == total_pages

        # Build era/context constraint block
        era_block = ""
        if context_constraints:
            era_block = f"""
        HISTORICAL/SETTING CONSTRAINTS (CRITICAL - MUST BE FOLLOWED):
        {context_constraints}

        ALL visual descriptions, costumes, technology, vehicles, and props MUST conform to these constraints.
        """

        # Build continuity block for non-first chunks
        continuity_block = ""
        if previous_summary:
            continuity_block = f"""
        STORY SO FAR (Pages 1-{start_page - 1}):
        {previous_summary}

        CONTINUE the story from where it left off. Maintain character and plot consistency.
        """

        # Pacing rules vary by chunk position
        if is_first_chunk and is_last_chunk:
            pacing_rules = f"""
        - Page 1 MUST introduce the setting/protagonist.
        - Page {total_pages} MUST contain the ending or a major cliffhanger.
        - Distribute the story arc evenly."""
        elif is_first_chunk:
            pacing_rules = f"""
        - Page 1 MUST introduce the setting/protagonist.
        - End this chunk with rising action or a plot development.
        - This is pages {start_page}-{end_page} of a {total_pages}-page novel."""
        elif is_last_chunk:
            pacing_rules = f"""
        - Build toward the climax in the middle pages.
        - Page {total_pages} MUST contain the ending or resolution.
        - This is the FINAL chunk (pages {start_page}-{end_page})."""
        else:
            pacing_rules = f"""
        - Continue the rising action and develop the plot.
        - End with a scene that leads into the next section.
        - This is pages {start_page}-{end_page} of a {total_pages}-page novel."""

        prompt = f"""
        Act as a Master Graphic Novel Director with deep understanding of comic book pacing.

        TASK:
        Generate pages {start_page} to {end_page} of a {total_pages}-PAGE Graphic Novel adaptation.
        {era_block}
        {continuity_block}
        {beat_guidance}
        {adaptation_guidance}
        You must output a JSON list of exactly {chunk_size} items.
        Each item represents ONE PAGE and must define:
        1. 'page_number': Integer ({start_page} to {end_page}).
        2. 'summary': A 2-sentence summary of what happens on this page.
        3. 'focus_text': 2-3 KEY VISUAL MOMENTS from the source text (not a 200-word excerpt).
           Focus on moments that can be DRAWN: actions, expressions, settings.
        4. 'mood': The emotional tone (e.g., "Tense", "Melancholic").
        5. 'key_characters': List of characters present.
        6. 'visual_notes': Specific setting or lighting notes. Include era-appropriate details.
        7. 'scene_type': One of "action", "dialogue", "establishing", "montage", "flashback", "transition", "underwater", "formal"

        NEW - SPREAD & PACING FIELDS:
        8. 'is_spread': Boolean - true if this page should be a TWO-PAGE SPREAD (panoramic reveal, epic battle).
           Use sparingly! Only 1-3 spreads per 50 pages. Spreads MUST be on EVEN page numbers (left page).
        9. 'is_cliffhanger': Boolean - true if this page ends with a hook/reveal that makes readers turn the page.
           IMPORTANT: Cliffhangers should land on ODD pages (right side) so the reveal is on the next turn.
        10. 'page_turn_note': Brief guidance on reader experience at page turn (e.g., "Reveal the monster on next page",
            "Pause before the decision", "Immediate action continues").
        11. 'suggested_panel_count': Integer 3-6. Fewer panels = more dramatic weight per image.
            Use 2-3 for climactic moments, 5-6 for rapid action/transitions.
        12. 'recommended_splash': Boolean - true if one panel should dominate (60%+ of page).

        SPREAD RULES:
        - Spreads work best for: vast landscapes, epic reveals, battle panoramas, emotional climaxes
        - A spread uses pages N (even) and N+1 (odd) as ONE image
        - Mark ONLY the first page of the spread as is_spread=true
        - The page AFTER a spread should be a quieter moment (pacing reset)

        CLIFFHANGER RULES:
        - Best cliffhangers: shocking reveals, danger moments, emotional decisions, mysterious arrivals
        - Place cliffhangers on ODD page numbers (right side of spread when book is open)
        - The NEXT page (even, left side) is what readers see first when turning

        CRITICAL PACING RULES:
        {pacing_rules}

        STYLE: {style}

        OUTPUT FORMAT: JSON List of {chunk_size} page objects.
        """

        model = config.scripting_model_global_context

        if cache_name:
            contents = [prompt]
            cached_content = cache_name
        else:
            contents = [prompt, f"SOURCE BOOK:\n{full_text_fallback}"]
            cached_content = None

        # Acquire TPM capacity
        blueprint_estimated = estimate_tokens_for_text(prompt)
        await get_tpm_limiter().acquire(blueprint_estimated)

        response = await get_client().aio.models.generate_content(
            model=model,
            contents=contents,
            config=types.GenerateContentConfig(
                cached_content=cached_content,
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
                            "visual_notes": {"type": "STRING"},
                            "scene_type": {"type": "STRING"},
                            # New spread and pacing fields
                            "is_spread": {"type": "BOOLEAN"},
                            "is_cliffhanger": {"type": "BOOLEAN"},
                            "page_turn_note": {"type": "STRING"},
                            "suggested_panel_count": {"type": "INTEGER"},
                            "recommended_splash": {"type": "BOOLEAN"}
                        },
                        "required": ["page_number", "summary", "focus_text", "key_characters", "scene_type"]
                    }
                }
            )
        )

        # Update TPM with actual usage
        bp_input, bp_output = extract_token_usage(response)
        get_tpm_limiter().update_actual_usage(blueprint_estimated, bp_input + bp_output)

        return response.parsed

    async def generate_pacing_blueprint(self, cache_name: str, full_text_fallback: str, target_pages: int, style: str, context_constraints: str = "", beat_map: dict = None, adaptation_filter: dict = None):
        """
        PASS 2: THE DIRECTOR
        Consumes the FULL BOOK (via cache) and outputs a page-by-page blueprint.

        Enhanced to use beat_map for intelligent page allocation and adds scene_type.
        Now also uses adaptation_filter to know what to emphasize/cut.

        For large jobs (>40 pages), uses chunked generation to avoid API timeouts
        while maximizing use of Gemini's context cache.

        Args:
            cache_name: Gemini context cache reference
            full_text_fallback: Full text if cache unavailable
            target_pages: Target number of pages
            style: Art style
            context_constraints: Era/setting constraints
            beat_map: Optional beat analysis from Pass 1 for pacing guidance
            adaptation_filter: Optional adaptation filter from Pass 1.5
        """
        print(f"🎬 DIRECTOR PASS: Creating {target_pages}-page blueprint...")

        # Build beat guidance if available
        beat_guidance = ""
        if beat_map and beat_map.get('beats'):
            beats = beat_map['beats']
            beat_list = "\n".join([
                f"  Beat {b['beat_id']}: {b['description']} ({b.get('page_allocation', 1)} pages, {b.get('beat_type', 'rising')}, visual: {b.get('visual_potential', 0.5):.1f})"
                for b in beats
            ])
            beat_guidance = f"""
        NARRATIVE BEAT GUIDANCE (from story analysis):
        Allocate pages according to this beat structure:
{beat_list}

        Use the page allocation to determine pacing. High-intensity beats (climax, crisis) get more pages.
        Prioritize beats with high visual_potential scores.
        """
            print(f"   Using beat-based allocation from {len(beats)} beats")

        # Build adaptation guidance if available
        adaptation_guidance = ""
        if adaptation_filter:
            # Essential scenes to keep
            essential = adaptation_filter.get('essential_scenes', [])
            if essential:
                essential_list = "\n".join([f"  - {s['description']}" for s in essential[:5]])
                adaptation_guidance += f"""
        MUST-KEEP SCENES (essential to the story):
{essential_list}
        """

            # Scenes to condense
            condensable = adaptation_filter.get('condensable_scenes', [])
            if condensable:
                condensable_list = "\n".join([f"  - {s['description']} → {s['condensation_strategy']}" for s in condensable[:5]])
                adaptation_guidance += f"""
        CONDENSE THESE SCENES (use suggested strategy):
{condensable_list}
        """

            # Reader-beloved moments
            beloved = adaptation_filter.get('reader_beloved_moments', [])
            if beloved:
                beloved_list = "\n".join([f"  - {s['description']}" for s in beloved])
                adaptation_guidance += f"""
        FAN-FAVORITE MOMENTS (must include, make visually stunning):
{beloved_list}
        """

            # Pacing recommendations
            pacing = adaptation_filter.get('pacing_recommendations', {})
            slow_down = pacing.get('slow_down', [])
            speed_up = pacing.get('speed_up', [])
            if slow_down or speed_up:
                adaptation_guidance += f"""
        PACING NOTES:
        - Slow down for beats: {slow_down if slow_down else 'none'}
        - Speed through beats: {speed_up if speed_up else 'none'}
        """
            print(f"   Using adaptation filter (essential: {len(essential)}, beloved: {len(beloved)})")

        # Determine if we need chunked generation
        if target_pages <= self.CHUNK_THRESHOLD:
            # Small job: single request with dynamic timeout
            timeout = calculate_dynamic_timeout(target_pages)
            print(f"   Using single-request mode (timeout: {timeout}s)")

            blueprint = await retry_api_call(
                self._generate_blueprint_chunk,
                cache_name,
                full_text_fallback,
                1,  # start_page
                target_pages,  # end_page
                target_pages,  # total_pages
                style,
                context_constraints,
                "",  # no previous summary
                beat_guidance,  # beat guidance from analysis
                adaptation_guidance,  # adaptation filter guidance
                timeout_seconds=timeout
            )
            return blueprint

        # Large job: chunked generation
        chunks = []
        start_page = 1

        while start_page <= target_pages:
            end_page = min(start_page + self.CHUNK_SIZE - 1, target_pages)
            chunks.append((start_page, end_page))
            start_page = end_page + 1

        print(f"   Using chunked mode: {len(chunks)} chunks of ~{self.CHUNK_SIZE} pages each")

        full_blueprint = []
        previous_summary = ""

        for i, (chunk_start, chunk_end) in enumerate(chunks):
            chunk_size = chunk_end - chunk_start + 1
            timeout = calculate_dynamic_timeout(chunk_size)

            print(f"   📄 Generating chunk {i+1}/{len(chunks)} (pages {chunk_start}-{chunk_end}, timeout: {timeout}s)...")

            chunk_blueprint = await retry_api_call(
                self._generate_blueprint_chunk,
                cache_name,
                full_text_fallback,
                chunk_start,
                chunk_end,
                target_pages,
                style,
                context_constraints,
                previous_summary,
                beat_guidance,  # beat guidance from analysis
                adaptation_guidance,  # adaptation filter guidance
                timeout_seconds=timeout
            )

            if not chunk_blueprint:
                raise ValueError(f"Failed to generate blueprint chunk {i+1} (pages {chunk_start}-{chunk_end})")

            full_blueprint.extend(chunk_blueprint)

            # Build enhanced summary for next chunk (last 8 pages with more detail)
            recent_pages = chunk_blueprint[-min(8, len(chunk_blueprint)):]
            previous_summary = "\n".join([
                f"Page {p['page_number']}: {p['summary']} (Characters: {', '.join(p.get('key_characters', []))}, Mood: {p.get('mood', 'neutral')}, Scene: {p.get('scene_type', 'dialogue')})"
                for p in recent_pages
            ])

            print(f"   ✅ Chunk {i+1} complete ({len(chunk_blueprint)} pages)")

        # Validate page numbers are sequential
        full_blueprint.sort(key=lambda x: x['page_number'])

        print(f"✅ Blueprint complete: {len(full_blueprint)} pages generated")
        return full_blueprint

    @retry_with_backoff()
    async def write_page_script(self, blueprint_item: dict, style: str, context_constraints: str = "", character_arcs: dict = None, assets: dict = None):
        """
        PASS 5: THE SCRIPTWRITER (Enhanced)
        Generates the detailed panel script for a SINGLE page, strictly following the blueprint.
        Uses character_arcs for scene-specific gear and emotional states.
        Uses assets for interaction_rules.
        """
        page_num = blueprint_item['page_number']
        focus_text = blueprint_item.get('focus_text', "")
        scene_type = blueprint_item.get('scene_type', 'dialogue')

        # Build era/context constraint block if provided
        era_block = ""
        if context_constraints:
            era_block = f"""
            HISTORICAL/SETTING CONSTRAINTS (CRITICAL):
            {context_constraints}

            ALL visual descriptions MUST conform to these constraints. NO anachronisms.
            """

        # Build scene-specific gear context from character_arcs
        scene_context = ""
        if character_arcs:
            # Find scene state for this page (or nearest earlier page)
            scene_states = character_arcs.get('scene_states', [])
            relevant_state = None
            for state in sorted(scene_states, key=lambda x: x.get('page_number', 0), reverse=True):
                if state.get('page_number', 0) <= page_num:
                    relevant_state = state
                    break

            if relevant_state:
                char_states = []
                for char_name, char_state in relevant_state.get('characters', {}).items():
                    gear = ', '.join(char_state.get('gear', []))
                    emotion = char_state.get('emotional_state', 'neutral')
                    char_states.append(f"  - {char_name}: {emotion}, gear: [{gear}]")

                rules = relevant_state.get('interaction_rules', [])
                rules_text = '\n'.join(f"  - {r}" for r in rules) if rules else "  (none specified)"

                scene_context = f"""
            CHARACTER STATES FOR THIS SCENE (Page {relevant_state.get('page_number')}):
{chr(10).join(char_states)}

            INTERACTION RULES:
{rules_text}
            """

        # Build interaction rules from assets
        interaction_context = ""
        if assets and assets.get('interaction_rules'):
            rules = assets['interaction_rules']
            scene_rules = rules.get(f'{scene_type}_scenes', [])
            if scene_rules:
                interaction_context = f"""
            SCENE TYPE: {scene_type}
            REQUIRED FOR THIS SCENE TYPE:
            {chr(10).join(f'  - {r}' for r in scene_rules)}
            """

        # Extract blueprint pacing hints
        suggested_panel_count = blueprint_item.get('suggested_panel_count', 4)
        recommended_splash = blueprint_item.get('recommended_splash', False)
        is_spread = blueprint_item.get('is_spread', False)
        is_cliffhanger = blueprint_item.get('is_cliffhanger', False)

        # Build location context from assets if available
        location_context = ""
        if assets and assets.get('locations'):
            # Find relevant location based on scene type and visual notes
            visual_notes = blueprint_item.get('visual_notes', '').lower()
            for loc in assets['locations']:
                loc_name = loc.get('name', '').lower()
                if loc_name in visual_notes or any(elem.lower() in visual_notes for elem in loc.get('recurring_elements', [])):
                    location_context = f"""
            LOCATION: {loc.get('name')}
            - Lighting: {loc.get('lighting')}
            - Mood: {loc.get('mood')}
            - MUST include: {', '.join(loc.get('recurring_elements', [])[:5])}
            - Color palette: {', '.join(loc.get('color_palette', [])[:3])}
            """
                    break

        async with get_scribe_limiter():
            print(f"✍️  Scripting Page {page_num} ({scene_type}, {suggested_panel_count} panels)...")

            prompt = f"""
            Act as a Graphic Novel Scriptwriter with expertise in CINEMATIC VISUAL STORYTELLING.

            TASK:
            Write the panel-by-panel script for PAGE {page_num}.
            {era_block}
            {scene_context}
            {interaction_context}
            {location_context}
            BLUEPRINT FOR THIS PAGE:
            Summary: {blueprint_item['summary']}
            Mood: {blueprint_item['mood']}
            Visual Notes: {blueprint_item.get('visual_notes', '')}
            Scene Type: {scene_type}
            Suggested Panel Count: {suggested_panel_count}
            Recommended Splash Panel: {recommended_splash}
            Is Two-Page Spread: {is_spread}
            Is Cliffhanger: {is_cliffhanger}

            STYLE: {style}

            SOURCE TEXT FOR THIS PAGE:
            "{focus_text}"

            INSTRUCTIONS:
            1. Break this page into {suggested_panel_count} panels {'(with one LARGE splash panel)' if recommended_splash else ''}.
            2. Use 'visual_description' for the artist (cinematic, detailed, era-appropriate).
            3. CRITICAL - For EACH panel, specify:
               - 'shot_type': One of:
                 * "establishing" - Wide shot showing full environment/setting
                 * "wide" - Full body shots, action scenes, multiple characters
                 * "medium" - Waist-up, good for dialogue and interaction
                 * "close-up" - Head and shoulders, emotional moments, reveals
                 * "extreme-close-up" - Eyes, hands, details - maximum drama
                 * "over-shoulder" - Looking over one character at another
                 * "two-shot" - Two characters framed together
                 * "birds-eye" - Looking down from above
                 * "worms-eye" - Looking up from below (heroic/imposing)
               - 'panel_size': One of "large" (50%+ of page), "medium" (25-50%), "small" (15-25%)
                 * Use "large" for dramatic reveals, action climaxes, establishing shots
                 * Use "small" for rapid action sequences, reactions, transitions
            4. Use 'dialogue' for character speech ONLY:
               - MAXIMUM 80 characters per dialogue field
               - Short, punchy speech bubbles only
               - NO stage directions, NO sound effects
               - If dialogue is interrupted, use em-dash: "Wait—"
            5. Use 'caption' for narration boxes (MAXIMUM 100 chars)
            6. For EACH panel, provide a STRUCTURED 'advice' object with:
               - 'scene_type': "{scene_type}"
               - 'required_gear': Object mapping character names to gear list
               - 'era_constraints': Era-specific requirements
               - 'continuity': {{'from_previous': "...", 'to_next': "..."}}
               - 'composition': {{'negative_space': "top-left|top-right|bottom-left|bottom-right"}}

            PACING RULES:
            {'- This page should have ONE dominant splash panel (large) with supporting smaller panels.' if recommended_splash else ''}
            {'- This is a TWO-PAGE SPREAD - the image extends across both pages. Make it EPIC.' if is_spread else ''}
            {'- This page ends on a CLIFFHANGER - the final panel should be dramatic/tense!' if is_cliffhanger else ''}
            - Vary shot types for visual interest (don't use same shot type consecutively)
            - Start scenes with establishing/wide shots, then move closer for emotional beats
            - Use close-ups for reactions and extreme-close-ups sparingly for maximum impact

            OUTPUT FORMAT: JSON.
            """

            # Prepare request - NO CACHED CONTENT used here to save TPM in parallel calls
            model = config.scripting_model_page_script
            contents = [prompt]

            # Acquire TPM capacity for page script generation
            script_estimated = estimate_tokens_for_text(prompt)
            await get_tpm_limiter().acquire(script_estimated)

            response = await get_client().aio.models.generate_content(
                model=model,
                contents=contents,
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
                                        "characters": {"type": "ARRAY", "items": {"type": "STRING"}},
                                        "key_objects": {"type": "ARRAY", "items": {"type": "STRING"}},
                                        "bubble_position": {"type": "STRING", "enum": ["top-left", "top-right", "bottom-left", "bottom-right", "caption-box"]},
                                        # New cinematic fields
                                        "shot_type": {
                                            "type": "STRING",
                                            "enum": ["establishing", "wide", "medium", "close-up", "extreme-close-up", "over-shoulder", "two-shot", "birds-eye", "worms-eye"]
                                        },
                                        "panel_size": {
                                            "type": "STRING",
                                            "enum": ["large", "medium", "small"]
                                        },
                                        "advice": {
                                            "type": "OBJECT",
                                            "properties": {
                                                "scene_type": {"type": "STRING"},
                                                "required_gear": {
                                                    "type": "OBJECT",
                                                    "additionalProperties": {
                                                        "type": "ARRAY",
                                                        "items": {"type": "STRING"}
                                                    }
                                                },
                                                "era_constraints": {"type": "ARRAY", "items": {"type": "STRING"}},
                                                "continuity": {
                                                    "type": "OBJECT",
                                                    "properties": {
                                                        "from_previous": {"type": "STRING"},
                                                        "to_next": {"type": "STRING"}
                                                    }
                                                },
                                                "composition": {
                                                    "type": "OBJECT",
                                                    "properties": {
                                                        "negative_space": {"type": "STRING"}
                                                    }
                                                }
                                            }
                                        }
                                    },
                                    "required": ["panel_id", "visual_description", "characters", "shot_type", "panel_size", "advice"]
                                }
                            }
                        },
                        "required": ["page_number", "panels"]
                    }
                )
            )

            # Update TPM with actual usage
            script_input, script_output = extract_token_usage(response)
            get_tpm_limiter().update_actual_usage(script_estimated, script_input + script_output)

            # Post-process: Ensure page number matches and enforce text limits
            result = response.parsed
            if not result:
                raise ValueError(f"Failed to parse script for page {page_num}")
                
            result['page_number'] = page_num

            # Enforce text length limits (safety net if LLM doesn't follow instructions)
            MAX_DIALOGUE_CHARS = 100
            MAX_CAPTION_CHARS = 120

            for panel in result.get('panels', []):
                # Truncate dialogue if too long
                dialogue = panel.get('dialogue', '')
                if dialogue:
                    # Clean up bad punctuation patterns
                    dialogue = re.sub(r'\.\.\.!', '—', dialogue)  # ...! -> em-dash
                    dialogue = re.sub(r'\?\!', '?', dialogue)  # ?! -> ?
                    dialogue = re.sub(r'\!\?', '!', dialogue)  # !? -> !
                    dialogue = re.sub(r'\.\.\.\.+', '...', dialogue)  # Multiple dots -> three
                    dialogue = re.sub(r'!!+', '!', dialogue)  # Multiple ! -> one
                    dialogue = re.sub(r'\?\?+', '?', dialogue)  # Multiple ? -> one

                    if len(dialogue) > MAX_DIALOGUE_CHARS:
                        # Find last word boundary before limit
                        truncated = dialogue[:MAX_DIALOGUE_CHARS].rsplit(' ', 1)[0]
                        dialogue = truncated + '...'
                    panel['dialogue'] = dialogue

                # Truncate caption if too long
                caption = panel.get('caption', '')
                if caption and len(caption) > MAX_CAPTION_CHARS:
                    truncated = caption[:MAX_CAPTION_CHARS].rsplit(' ', 1)[0]
                    panel['caption'] = truncated + '...'

            return result

    async def generate_script(self, style: str, test_mode=True, context_constraints: str = "", target_page_override: int = None):
        """
        Main orchestration method - 7-pass enrichment pipeline:

        PASS 0: Global Context (provided via context_constraints)
        PASS 1: Beat Analysis - Extract narrative beats, intensity, act structure
        PASS 2: Director Pass - Blueprint with beat-based page allocation
        PASS 3: Character Deep Dive - Arcs, relationships, scene-specific gear
        PASS 4: Asset Manifest - Characters, objects, interaction rules
        PASS 5: Scriptwriter Pass - Panel scripts with structured advice
        PASS 6: Validation + Auto-Fix - Validate and fix common issues
        """
        from validators import validate_and_autofix_script, ValidationReport

        # Output paths
        suffix = "_test_page.json" if test_mode else "_full_script.json"
        output_file = self.output_dir / f"{self.book_path.stem}{suffix}"

        if output_file.exists() and not test_mode:
             print(f"⏭️  Skipping Script Generation (Already exists: {output_file})")
             with open(output_file, "r") as f:
                 return json.load(f)

        # Load Content
        full_text = self.load_content(test_mode=False)  # Always load full for caching
        target_pages = target_page_override or (3 if test_mode else 10)

        print(f"\n📚 Scripting '{self.book_path.stem}'")
        print(f"   Target Length: {target_pages} pages")
        print(f"   Style: {style}")

        # Create Cache (The Enabler)
        cache_name = await self._get_or_create_cache(full_text)

        # Log context constraints if provided
        if context_constraints:
            print(f"   Era/Context: {context_constraints[:100]}...")

        # ========== PASS 1: Beat Analysis ==========
        print("\n🎭 PASS 1: Analyzing narrative beats...")
        beat_map = await self.analyze_narrative_beats(cache_name, full_text, target_pages)

        # Save beat map
        beats_path = self.output_dir / f"{self.book_path.stem}_beats.json"
        with open(beats_path, "w") as f:
            json.dump(beat_map, f, indent=2)
        print(f"✅ Beat analysis complete: {beats_path}")
        print(f"   Found {len(beat_map.get('beats', []))} narrative beats")

        # ========== PASS 1.5: Adaptation Filter ==========
        print("\n✂️  PASS 1.5: Running adaptation filter...")
        adaptation_filter = await self.generate_adaptation_filter(
            cache_name, full_text, beat_map, target_pages
        )

        # Save adaptation filter
        filter_path = self.output_dir / f"{self.book_path.stem}_adaptation.json"
        with open(filter_path, "w") as f:
            json.dump(adaptation_filter, f, indent=2)
        print(f"✅ Adaptation filter complete: {filter_path}")

        # ========== PASS 2: Director Pass (Blueprint) ==========
        print("\n🎬 PASS 2: Generating pacing blueprint...")
        blueprint = await self.generate_pacing_blueprint(
            cache_name, full_text, target_pages, style, context_constraints, beat_map, adaptation_filter
        )

        # Save blueprint
        blueprint_path = self.output_dir / f"{self.book_path.stem}_blueprint.json"
        with open(blueprint_path, "w") as f:
            json.dump(blueprint, f, indent=2)
        print(f"✅ Blueprint created: {blueprint_path}")

        # ========== PASS 3: Character Deep Dive ==========
        print("\n👥 PASS 3: Generating character deep dive...")
        character_arcs = await self.generate_character_deep_dive(
            cache_name, full_text, blueprint
        )

        # Save character arcs
        arcs_path = self.output_dir / f"{self.book_path.stem}_character_arcs.json"
        with open(arcs_path, "w") as f:
            json.dump(character_arcs, f, indent=2)
        print(f"✅ Character arcs created: {arcs_path}")
        print(f"   Tracking {len(character_arcs.get('characters', []))} characters")

        # ========== PASS 4: Asset Manifest ==========
        print("\n📋 PASS 4: Generating asset manifest...")
        asset_manifest = await self.generate_asset_manifest(
            cache_name, full_text, blueprint, style, context_constraints, character_arcs
        )

        # Save asset manifest
        manifest_path = self.output_dir / f"{self.book_path.stem}_assets.json"
        with open(manifest_path, "w") as f:
            json.dump(asset_manifest, f, indent=2)
        print(f"✅ Asset manifest created: {manifest_path}")

        # ========== PASS 5: Scriptwriter Pass (Parallel) ==========
        print(f"\n⚡ PASS 5: Generating panel scripts for {len(blueprint)} pages...")

        tasks = [
            self.write_page_script(item, style, context_constraints, character_arcs, asset_manifest)
            for item in blueprint
        ]

        # Execute all pages at once (bounded by get_scribe_limiter())
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Separate successful scripts from failures
        full_script = []
        failed_pages = []
        failed_blueprints = {}  # Map page_num to blueprint item for retry

        for i, result in enumerate(results):
            if isinstance(result, Exception):
                page_num = blueprint[i].get('page_number', i + 1)
                print(f"   ❌ Page {page_num} script failed: {result}")
                failed_pages.append(page_num)
                failed_blueprints[page_num] = blueprint[i]
            else:
                full_script.append(result)

        # RETRY LOGIC: Attempt to regenerate failed pages (up to 2 retries)
        MAX_RETRIES = 2
        retry_count = 0

        while failed_pages and retry_count < MAX_RETRIES:
            retry_count += 1
            print(f"\n🔄 Retry attempt {retry_count}/{MAX_RETRIES} for {len(failed_pages)} failed page(s)...")

            # Create retry tasks for failed pages
            retry_tasks = [
                self.write_page_script(
                    failed_blueprints[page_num],
                    style,
                    context_constraints,
                    character_arcs,
                    asset_manifest
                )
                for page_num in failed_pages
            ]

            # Execute retry tasks with a small delay between each
            retry_results = await asyncio.gather(*retry_tasks, return_exceptions=True)

            # Process retry results
            still_failed = []
            for i, result in enumerate(retry_results):
                page_num = failed_pages[i]
                if isinstance(result, Exception):
                    print(f"   ❌ Page {page_num} retry {retry_count} failed: {result}")
                    still_failed.append(page_num)
                else:
                    print(f"   ✅ Page {page_num} succeeded on retry {retry_count}")
                    full_script.append(result)

            failed_pages = still_failed

        if failed_pages:
            print(f"⚠️ {len(failed_pages)} page(s) still failed after {MAX_RETRIES} retries: {failed_pages}")
            print("   These pages will be missing from the final script. Consider re-running the pipeline.")

        # Sort by page number
        full_script.sort(key=lambda x: x['page_number'])

        # ========== PASS 6: Validation + Auto-Fix ==========
        print("\n🔍 PASS 6: Validating and auto-fixing script...")

        # Extract era from context_constraints for validation
        era = ""
        if context_constraints:
            # Try to extract era from constraints (e.g., "Setting: 1860s...")
            if "1860" in context_constraints or "Victorian" in context_constraints:
                era = "1860s Victorian"
            elif "1920" in context_constraints or "Art Deco" in context_constraints:
                era = "1920s Art Deco"
            elif "Medieval" in context_constraints or "medieval" in context_constraints:
                era = "Medieval Fantasy"

        fixed_script, validation_report = validate_and_autofix_script(
            script=full_script,
            era=era,
            character_arcs=character_arcs,
            assets=asset_manifest
        )

        # Save validation report
        validation_path = self.output_dir / f"{self.book_path.stem}_validation.json"
        with open(validation_path, "w") as f:
            report_dict = {
                "total_issues": validation_report.total_issues,
                "auto_fixed": validation_report.auto_fixed,
                "warnings": validation_report.warnings,
                "manual_review": validation_report.manual_review,
                "script_modified": validation_report.script_modified,
                "issues": [
                    {
                        "issue_type": issue.issue_type,
                        "page": issue.page,
                        "panel": issue.panel,
                        "severity": issue.severity,
                        "description": issue.description,
                        "fix_applied": issue.fix_applied,
                        "original_value": issue.original_value,
                        "fixed_value": issue.fixed_value
                    }
                    for issue in validation_report.issues
                ]
            }
            json.dump(report_dict, f, indent=2)

        print(f"✅ Validation complete: {validation_path}")
        print(f"   Total issues: {validation_report.total_issues}")
        print(f"   Auto-fixed: {validation_report.auto_fixed}")
        print(f"   Manual review needed: {validation_report.manual_review}")

        # Use fixed script if modifications were made
        if validation_report.script_modified:
            full_script = fixed_script
            print("   Applied auto-fixes to script")

        # Save Final Script
        with open(output_file, "w") as f:
            json.dump(full_script, f, indent=2)

        print(f"\n🎉 Script Generation Complete! Saved to: {output_file}")
        return full_script

if __name__ == "__main__":
    # Test run
    agent = ScriptingAgent("assets/input/20-thousand-leagues-under-the-sea.txt")
    asyncio.run(agent.generate_script(
        style="Lush Watercolor", 
        test_mode=True, 
        target_page_override=3
    ))