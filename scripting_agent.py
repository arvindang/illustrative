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

from google import genai
from google.genai import types
from utils import (
    retry_with_backoff, RateLimiter, get_tpm_limiter,
    estimate_tokens_for_text, estimate_tokens_for_cache, extract_token_usage,
    retry_api_call, calculate_dynamic_timeout, calculate_beat_density,
    fix_era_anachronisms
)
from config import config

# Lazy client initialization for runtime API key support
_client = None
_client_key = None

def get_client():
    """Returns a Gemini client, creating a new one if API key has changed."""
    global _client, _client_key
    if _client is None or _client_key != config.gemini_api_key:
        _client = genai.Client(api_key=config.gemini_api_key)
        _client_key = config.gemini_api_key
    return _client

# Rate limiter for the parallel script writing phase
# We can go higher here because we aren't generating images yet
scribe_limiter = RateLimiter(rpm_limit=15)

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
        PASS 1: BEAT ANALYSIS
        Analyzes the narrative structure and identifies story beats.

        This pass breaks the novel into narrative beats with intensity scores,
        identifies act structure, and calculates page allocation.

        Args:
            cache_name: Gemini context cache reference
            full_text_fallback: Full text if cache unavailable
            target_pages: Target number of pages for allocation

        Returns:
            BeatMap dict with beats, act_boundaries, pacing_recommendations
        """
        print(f"🎭 BEAT ANALYSIS PASS: Extracting narrative structure...")

        prompt = f"""
        Act as a Story Structure Analyst for graphic novel adaptation.

        TASK:
        Analyze the source material and identify the key narrative BEATS.
        A beat is a significant story event that advances the plot or reveals character.

        For a {target_pages}-page graphic novel, identify approximately {max(5, target_pages // 5)} major beats.

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

        Also identify:
        - 'act_boundaries': Which beat_id ends Act 1 and Act 2 (for 3-act structure)
        - 'total_word_count': Approximate word count of source material

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
                                    "scene_type": {"type": "STRING"}
                                },
                                "required": ["beat_id", "beat_type", "description", "intensity", "key_characters"]
                            }
                        },
                        "act_boundaries": {
                            "type": "OBJECT",
                            "properties": {
                                "act_1_end": {"type": "INTEGER"},
                                "act_2_end": {"type": "INTEGER"}
                            }
                        }
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

        print(f"✅ Identified {len(beats)} narrative beats")
        print(f"   Act 1 ends at beat {beat_map.get('act_boundaries', {}).get('act_1_end', '?')}")
        print(f"   Act 2 ends at beat {beat_map.get('act_boundaries', {}).get('act_2_end', '?')}")

        return beat_map

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
                                    }
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

        # Combined prompt for characters and objects
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

        Characters to design: {', '.join(characters)}

        PART 2: KEY OBJECTS & LOCATIONS
        Identify the top 3-5 most important RECURRING OBJECTS, VEHICLES, or LOCATIONS (e.g., 'The Nautilus') that need consistent visual design.
        For each, provide:
        - name: Name of the object
        - description: Visual description (materials, textures, colors). Must be era-appropriate.
        - key_features: List of identifying shapes or mechanisms (era-appropriate technology)
        - condition: The state of wear (e.g., "Pristine and polished", "Rusted and barnacle-encrusted", "Ancient and crumbling")
        - material_context: Primary materials appropriate to the era (e.g., "Riveted iron and brass", "Bioluminescent organic matter")

        PART 3: INTERACTION RULES
        Define scene-specific requirements for visual consistency:
        - underwater_scenes: List requirements (e.g., "All characters MUST wear period diving equipment with brass helmets")
        - formal_scenes: List requirements (e.g., "Victorian evening wear, men in frock coats")
        - action_scenes: List requirements (e.g., "Practical clothing, freedom of movement")
        - aboard_ship: List requirements (e.g., "Nautical attire appropriate to role")

        Also identify any 'forbidden_combinations' - things that should NEVER appear together:
        - Example: {{"characters": ["Professor Aronnax"], "items": ["modern SCUBA gear"], "reason": "anachronism"}}

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
                                    "specific_era_markers": {"type": "STRING"}
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
                        }
                    },
                    "required": ["characters", "objects", "interaction_rules"]
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
        print(f"🛠️ Found {len(manifest['objects'])} key objects and {len(manifest['characters'])} characters.")
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
        beat_guidance: str = ""
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
        Act as a Master Graphic Novel Director.

        TASK:
        Generate pages {start_page} to {end_page} of a {total_pages}-PAGE Graphic Novel adaptation.
        {era_block}
        {continuity_block}
        {beat_guidance}
        You must output a JSON list of exactly {chunk_size} items.
        Each item represents ONE PAGE and must define:
        1. 'page_number': Integer ({start_page} to {end_page}).
        2. 'summary': A 2-sentence summary of what happens on this page.
        3. 'focus_text': A specific quote or 200-word excerpt from the source text that this page covers.
        4. 'mood': The emotional tone (e.g., "Tense", "Melancholic").
        5. 'key_characters': List of characters present.
        6. 'visual_notes': Specific setting or lighting notes. Include era-appropriate details.
        7. 'scene_type': One of "action", "dialogue", "establishing", "montage", "flashback", "transition", "underwater", "formal"

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
                            "scene_type": {"type": "STRING"}
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

    async def generate_pacing_blueprint(self, cache_name: str, full_text_fallback: str, target_pages: int, style: str, context_constraints: str = "", beat_map: dict = None):
        """
        PASS 2: THE DIRECTOR
        Consumes the FULL BOOK (via cache) and outputs a page-by-page blueprint.

        Enhanced to use beat_map for intelligent page allocation and adds scene_type.

        For large jobs (>40 pages), uses chunked generation to avoid API timeouts
        while maximizing use of Gemini's context cache.

        Args:
            cache_name: Gemini context cache reference
            full_text_fallback: Full text if cache unavailable
            target_pages: Target number of pages
            style: Art style
            context_constraints: Era/setting constraints
            beat_map: Optional beat analysis from Pass 1 for pacing guidance
        """
        print(f"🎬 DIRECTOR PASS: Creating {target_pages}-page blueprint...")

        # Build beat guidance if available
        beat_guidance = ""
        if beat_map and beat_map.get('beats'):
            beats = beat_map['beats']
            beat_list = "\n".join([
                f"  Beat {b['beat_id']}: {b['description']} ({b.get('page_allocation', 1)} pages, {b.get('beat_type', 'rising')})"
                for b in beats
            ])
            beat_guidance = f"""
        NARRATIVE BEAT GUIDANCE (from story analysis):
        Allocate pages according to this beat structure:
{beat_list}

        Use the page allocation to determine pacing. High-intensity beats (climax, crisis) get more pages.
        """
            print(f"   Using beat-based allocation from {len(beats)} beats")

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
                timeout_seconds=timeout
            )

            if not chunk_blueprint:
                raise ValueError(f"Failed to generate blueprint chunk {i+1} (pages {chunk_start}-{chunk_end})")

            full_blueprint.extend(chunk_blueprint)

            # Build summary for next chunk (last 3-5 pages of current chunk)
            recent_pages = chunk_blueprint[-min(5, len(chunk_blueprint)):]
            previous_summary = " | ".join([
                f"Page {p['page_number']}: {p['summary']}"
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

        async with scribe_limiter:
            print(f"✍️  Scripting Page {page_num} ({scene_type})...")

            prompt = f"""
            Act as a Graphic Novel Scriptwriter.

            TASK:
            Write the panel-by-panel script for PAGE {page_num}.
            {era_block}
            {scene_context}
            {interaction_context}
            BLUEPRINT FOR THIS PAGE:
            Summary: {blueprint_item['summary']}
            Mood: {blueprint_item['mood']}
            Visual Notes: {blueprint_item.get('visual_notes', '')}
            Scene Type: {scene_type}

            STYLE: {style}

            SOURCE TEXT FOR THIS PAGE:
            "{focus_text}"

            INSTRUCTIONS:
            1. Break this page into 3-6 panels based on the SOURCE TEXT.
            2. Use 'visual_description' for the artist (cinematic, detailed, era-appropriate).
            3. Use 'dialogue' for character speech ONLY:
               - MAXIMUM 80 characters per dialogue field
               - Short, punchy speech bubbles only
               - NO stage directions like (SFX), (Internal Monologue), (whispers)
               - NO sound effects in dialogue
               - Use proper punctuation: ONE exclamation OR question mark, never "...!" or "?!"
               - If dialogue is interrupted, use em-dash: "Wait—" NOT "Wait...!"
               - Each speech bubble should be ONE complete thought from ONE speaker
               - NO combining multiple speakers in one dialogue field
            4. Use 'caption' for narration boxes:
               - MAXIMUM 100 characters per caption
               - Brief, evocative narration only
               - Third-person perspective for narration
            5. If a character has internal thoughts, put them in 'caption' NOT 'dialogue'.
            6. Ensure visual continuity with the blueprint notes.
            7. Use 'key_objects' to list important recurring items/vehicles present (e.g., ['The Nautilus', 'Harpoon']).
            8. For EACH panel, provide a STRUCTURED 'advice' object with:
               - 'scene_type': "{scene_type}" (must match page scene type)
               - 'required_gear': Object mapping character names to list of required gear
               - 'era_constraints': List of era-specific requirements for this panel
               - 'continuity': Object with 'from_previous' and 'to_next' descriptions
               - 'composition': Object with 'negative_space' position for text overlay

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
                                    "required": ["panel_id", "visual_description", "characters", "advice"]
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

        # ========== PASS 2: Director Pass (Blueprint) ==========
        print("\n🎬 PASS 2: Generating pacing blueprint...")
        blueprint = await self.generate_pacing_blueprint(
            cache_name, full_text, target_pages, style, context_constraints, beat_map
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

        # Execute all pages at once (bounded by scribe_limiter)
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Separate successful scripts from failures
        full_script = []
        failed_pages = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                page_num = blueprint[i].get('page_number', i + 1)
                print(f"   ❌ Page {page_num} script failed: {result}")
                failed_pages.append(page_num)
            else:
                full_script.append(result)

        if failed_pages:
            print(f"⚠️ {len(failed_pages)} page(s) failed during scripting: {failed_pages}")

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