"""
PASS 1: Beat Analysis

Analyzes the narrative structure and identifies story beats with visual potential scoring.
This pass breaks the novel into narrative beats with intensity scores, identifies act structure,
calculates page allocation, and scores visual potential.
"""
from google.genai import types

from utils import (
    retry_with_backoff, get_tpm_limiter,
    estimate_tokens_for_text, extract_token_usage, calculate_beat_density,
    get_client
)
from config import config
from agents.scripting.schemas import BEAT_ANALYSIS_SCHEMA


@retry_with_backoff()
async def analyze_narrative_beats(
    cache_name: str,
    full_text_fallback: str,
    target_pages: int
) -> dict:
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
            response_schema=BEAT_ANALYSIS_SCHEMA
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
