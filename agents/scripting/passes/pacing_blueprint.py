"""
PASS 2: Pacing Blueprint (Director Pass)

Generates a page-by-page blueprint with spread awareness and cliffhanger placement.
Supports chunked generation for large jobs (40+ pages).
"""
from google.genai import types

from utils import (
    get_tpm_limiter, estimate_tokens_for_text, extract_token_usage,
    retry_api_call, calculate_dynamic_timeout, get_client
)
from config import config
from agents.scripting.schemas import BLUEPRINT_PAGE_SCHEMA

# Threshold for chunked generation (pages above this get split into chunks)
CHUNK_THRESHOLD = 40
CHUNK_SIZE = 30  # Pages per chunk for large jobs


async def _generate_blueprint_chunk(
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

    NEW - FULL-BLEED & PACING FIELDS (for DIGITAL graphic novel - readers see ONE page at a time):
    8. 'is_spread': Boolean - true if this page should be a FULL-BLEED single page (one dramatic image filling entire page).
       Use sparingly! Only 1-3 full-bleeds per 50 pages. Best for: epic reveals, panoramic vistas, emotional climaxes.
    9. 'is_cliffhanger': Boolean - true if this page ends with a hook/reveal that makes readers swipe to next page.
       Best cliffhangers: shocking reveals, danger moments, emotional decisions, mysterious arrivals.
    10. 'page_turn_note': Brief guidance on reader experience at page turn (e.g., "Reveal the monster on next page",
        "Pause before the decision", "Immediate action continues").
    11. 'suggested_panel_count': Integer 3-6. Fewer panels = more dramatic weight per image.
        Use 2-3 for climactic moments, 5-6 for rapid action/transitions.
        Set to 1 for full-bleed pages (is_spread=true).
    12. 'recommended_splash': Boolean - true if one panel should dominate (60%+ of page).
        Different from full-bleed: splash has one LARGE panel with smaller supporting panels.

    FULL-BLEED RULES (is_spread=true):
    - Creates a SINGLE image covering the entire page (no panel divisions)
    - Best for: vast landscapes, epic reveals, battle panoramas, emotional climaxes, major character introductions
    - Use sparingly for maximum impact (1-3 per 50 pages)
    - The page AFTER a full-bleed should usually be a quieter moment (pacing reset)
    - When is_spread=true, set suggested_panel_count=1

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
            response_schema=BLUEPRINT_PAGE_SCHEMA
        )
    )

    # Update TPM with actual usage
    bp_input, bp_output = extract_token_usage(response)
    get_tpm_limiter().update_actual_usage(blueprint_estimated, bp_input + bp_output)

    return response.parsed


async def generate_pacing_blueprint(
    cache_name: str,
    full_text_fallback: str,
    target_pages: int,
    style: str,
    context_constraints: str = "",
    beat_map: dict = None,
    adaptation_filter: dict = None
) -> list:
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

    Returns:
        List of page blueprint dictionaries
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
    if target_pages <= CHUNK_THRESHOLD:
        # Small job: single request with dynamic timeout
        timeout = calculate_dynamic_timeout(target_pages)
        print(f"   Using single-request mode (timeout: {timeout}s)")

        blueprint = await retry_api_call(
            _generate_blueprint_chunk,
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
        end_page = min(start_page + CHUNK_SIZE - 1, target_pages)
        chunks.append((start_page, end_page))
        start_page = end_page + 1

    print(f"   Using chunked mode: {len(chunks)} chunks of ~{CHUNK_SIZE} pages each")

    full_blueprint = []
    previous_summary = ""

    for i, (chunk_start, chunk_end) in enumerate(chunks):
        chunk_size = chunk_end - chunk_start + 1
        timeout = calculate_dynamic_timeout(chunk_size)

        print(f"   📄 Generating chunk {i+1}/{len(chunks)} (pages {chunk_start}-{chunk_end}, timeout: {timeout}s)...")

        chunk_blueprint = await retry_api_call(
            _generate_blueprint_chunk,
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
