"""
PASS 1.5: Adaptation Filter

Identifies what content to cut, condense, or keep for the graphic novel adaptation.
This pass acts as an editorial filter, making creative decisions about scene importance.
"""
from google.genai import types

from utils import (
    retry_with_backoff, get_tpm_limiter,
    estimate_tokens_for_text, extract_token_usage,
    get_client
)
from config import config
from agents.scripting.schemas import ADAPTATION_FILTER_SCHEMA


@retry_with_backoff()
async def generate_adaptation_filter(
    cache_name: str,
    full_text_fallback: str,
    beat_map: dict,
    target_pages: int
) -> dict:
    """
    PASS 1.5: ADAPTATION FILTER
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
            response_schema=ADAPTATION_FILTER_SCHEMA
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
