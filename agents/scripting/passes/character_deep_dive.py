"""
PASS 3: Character Deep Dive

Generates detailed character arcs and scene-specific states.
Includes voice profiles for dialogue consistency.
"""
from typing import Callable

from google.genai import types

from utils import (
    retry_with_backoff, get_tpm_limiter,
    estimate_tokens_for_text, extract_token_usage,
    get_client
)
from config import config
from agents.scripting.schemas import CHARACTER_DEEP_DIVE_SCHEMA


@retry_with_backoff()
async def generate_character_deep_dive(
    cache_name: str,
    full_text_fallback: str,
    blueprint: list,
    extract_characters_fn: Callable[[list], list]
) -> dict:
    """
    PASS 3: CHARACTER DEEP DIVE
    Generates detailed character arcs and scene-specific states.

    Args:
        cache_name: Gemini context cache reference
        full_text_fallback: Full text if cache unavailable
        blueprint: Page-by-page blueprint from Pass 2
        extract_characters_fn: Function to extract characters from blueprint

    Returns:
        CharacterArcs dict with characters, scene_states
    """
    # Extract unique characters from blueprint
    characters = extract_characters_fn(blueprint)
    print(f"🎭 CHARACTER DEEP DIVE: Analyzing arcs for {len(characters)} characters...")

    # Create a summary of pages for context
    page_summaries = [
        f"Page {p['page_number']}: {p.get('summary', '')} (Characters: {', '.join(p.get('key_characters', []))})"
        for p in blueprint  # Use full blueprint so late-appearing characters get proper tracking
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
        contents = [prompt, f"SOURCE BOOK:\n{full_text_fallback[:config.fallback_text_max_chars]}"]
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
            response_schema=CHARACTER_DEEP_DIVE_SCHEMA
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
