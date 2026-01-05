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
    estimate_tokens_for_text, estimate_tokens_for_cache, extract_token_usage
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
    async def generate_asset_manifest(self, cache_name: str, full_text_fallback: str, blueprint: list, style: str, context_constraints: str = ""):
        """
        PASS 1.5: ASSET EXTRACTION
        Generates visual descriptions for characters and key objects in a single pass.
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

        # Combined prompt for characters and objects
        combined_prompt = f"""
        Act as a Visual Development Artist for a '{style}' graphic novel.
        {era_block}
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
                        }
                    },
                    "required": ["characters", "objects"]
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

    @retry_with_backoff()
    async def generate_pacing_blueprint(self, cache_name: str, full_text_fallback: str, target_pages: int, style: str, context_constraints: str = ""):
        """
        PASS 1: THE DIRECTOR
        Consumes the FULL BOOK (via cache) and outputs a page-by-page blueprint.
        """
        print(f"🎬 DIRECTOR PASS: Creating {target_pages}-page blueprint...")

        # Build era/context constraint block if provided
        era_block = ""
        if context_constraints:
            era_block = f"""
        HISTORICAL/SETTING CONSTRAINTS (CRITICAL - MUST BE FOLLOWED):
        {context_constraints}

        ALL visual descriptions, costumes, technology, vehicles, and props MUST conform to these constraints.
        Flag any elements that would be anachronistic or out-of-place.
        """

        prompt = f"""
        Act as a Master Graphic Novel Director.

        TASK:
        Adapt the provided book into a TIGHT {target_pages}-PAGE Graphic Novel Script.
        {era_block}
        You must output a JSON list of exactly {target_pages} items.
        Each item represents ONE PAGE and must define:
        1. 'page_number': Integer (1 to {target_pages}).
        2. 'summary': A 2-sentence summary of what happens on this page.
        3. 'focus_text': A specific quote or 200-word excerpt from the source text that this page covers.
        4. 'mood': The emotional tone (e.g., "Tense", "Melancholic").
        5. 'key_characters': List of characters present.
        6. 'visual_notes': Specific setting or lighting notes. Include era-appropriate details.

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

        # Acquire TPM capacity for blueprint generation
        blueprint_estimated = estimate_tokens_for_text(prompt)
        await get_tpm_limiter().acquire(blueprint_estimated)

        response = await get_client().aio.models.generate_content(
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

        # Update TPM with actual usage from blueprint generation
        bp_input, bp_output = extract_token_usage(response)
        get_tpm_limiter().update_actual_usage(blueprint_estimated, bp_input + bp_output)

        return response.parsed

    @retry_with_backoff()
    async def write_page_script(self, blueprint_item: dict, style: str, context_constraints: str = ""):
        """
        PASS 2: THE SCRIPTWRITER
        Generates the detailed panel script for a SINGLE page, strictly following the blueprint.
        Uses ONLY local context (focus_text) to minimize token consumption and avoid TPM limits.
        """
        page_num = blueprint_item['page_number']
        focus_text = blueprint_item.get('focus_text', "")

        # Build era/context constraint block if provided
        era_block = ""
        if context_constraints:
            era_block = f"""
            HISTORICAL/SETTING CONSTRAINTS (CRITICAL):
            {context_constraints}

            ALL visual descriptions MUST conform to these constraints. NO anachronisms.
            """

        async with scribe_limiter:
            print(f"✍️  Scripting Page {page_num}...")

            prompt = f"""
            Act as a Graphic Novel Scriptwriter.

            TASK:
            Write the panel-by-panel script for PAGE {page_num}.
            {era_block}
            BLUEPRINT FOR THIS PAGE:
            Summary: {blueprint_item['summary']}
            Mood: {blueprint_item['mood']}
            Visual Notes: {blueprint_item['visual_notes']}

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
            8. For EACH panel, provide an 'advice' object with:
               - 'historical_constraints': Era-specific requirements for this panel (clothing, tech, props)
               - 'continuity_notes': How this panel connects visually to adjacent panels
               - 'character_gear': What each character should be wearing/carrying in this panel
               - For UNDERWATER scenes: Characters MUST wear period diving suits (brass helmets, canvas suits)

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
                                                "historical_constraints": {"type": "STRING"},
                                                "continuity_notes": {"type": "STRING"},
                                                "character_gear": {"type": "STRING"}
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

        # Log context constraints if provided
        if context_constraints:
            print(f"   Era/Context: {context_constraints[:100]}...")

        # 3. Director Pass (Blueprint) - NOW WITH CONTEXT CONSTRAINTS
        blueprint = await self.generate_pacing_blueprint(cache_name, full_text, target_pages, style, context_constraints)

        # Save blueprint for debugging
        blueprint_path = self.output_dir / f"{self.book_path.stem}_blueprint.json"
        with open(blueprint_path, "w") as f:
            json.dump(blueprint, f, indent=2)
        print(f"✅ Blueprint created: {blueprint_path}")

        # 3.5. Asset Manifest (Characters & Objects) - NOW WITH CONTEXT CONSTRAINTS
        print("\n📋 Generating asset manifest...")
        asset_manifest = await self.generate_asset_manifest(cache_name, full_text, blueprint, style, context_constraints)

        # Save asset manifest for IllustratorAgent
        manifest_path = self.output_dir / f"{self.book_path.stem}_assets.json"
        with open(manifest_path, "w") as f:
            json.dump(asset_manifest, f, indent=2)
        print(f"✅ Asset manifest created: {manifest_path}")

        # 4. Scriptwriter Pass (Parallel) - NOW WITH CONTEXT CONSTRAINTS
        print(f"⚡ Starting Parallel Script Generation for {len(blueprint)} pages...")

        tasks = [
            self.write_page_script(item, style, context_constraints)
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