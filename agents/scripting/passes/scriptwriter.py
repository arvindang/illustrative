"""
PASS 5: Scriptwriter

Generates detailed panel scripts for each page, with support for cinematic
shot types, multiple dialogue bubbles, and scene continuity.
"""
import re

from google.genai import types

from utils import (
    retry_with_backoff, get_tpm_limiter,
    estimate_tokens_for_text, extract_token_usage,
    get_client
)
from config import config
from agents.scripting.schemas import PAGE_SCRIPT_SCHEMA
from agents.scripting.utils import get_scribe_limiter


@retry_with_backoff()
async def write_page_script(
    blueprint_item: dict,
    style: str,
    context_constraints: str = "",
    character_arcs: dict = None,
    assets: dict = None,
    prev_page: dict = None,
    next_page: dict = None,
    prev_page_script: dict = None
) -> dict:
    """
    PASS 5: THE SCRIPTWRITER (Enhanced)
    Generates the detailed panel script for a SINGLE page, strictly following the blueprint.

    Enhanced features:
    - Uses character_arcs for scene-specific gear, emotional states, AND voice profiles
    - Uses assets for interaction_rules and location context
    - Uses prev_page/next_page BLUEPRINTS for scene continuity
    - Uses prev_page_script (ACTUAL generated script) for dialogue flow continuity

    Args:
        blueprint_item: Blueprint data for this page
        style: Art style
        context_constraints: Era/setting constraints
        character_arcs: Character arc data with voice profiles
        assets: Asset manifest with locations and interaction rules
        prev_page: Blueprint of previous page (for continuity)
        next_page: Blueprint of next page (for setup/payoff)
        prev_page_script: ACTUAL generated script from previous page (for dialogue continuity)

    Returns:
        Page script dict with panels
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

    # Build voice profiles for characters on this page (for dialogue consistency)
    voice_context = ""
    page_characters = blueprint_item.get('key_characters', [])
    if character_arcs and page_characters:
        char_voices = []
        for char in character_arcs.get('characters', []):
            char_name = char.get('name', '')
            if char_name in page_characters:
                voice = char.get('voice_profile', {})
                samples = char.get('dialogue_samples', [])
                if voice or samples:
                    voice_info = f"  {char_name}:"
                    if voice.get('formality'):
                        voice_info += f" {voice.get('formality')} speech,"
                    if voice.get('vocabulary_style'):
                        voice_info += f" {voice.get('vocabulary_style')},"
                    if voice.get('dialect_markers'):
                        voice_info += f" markers: {', '.join(voice.get('dialect_markers', [])[:2])}"
                    if samples:
                        voice_info += f"\n    Example: \"{samples[0][:60]}...\""
                    char_voices.append(voice_info)

        if char_voices:
            voice_context = f"""
        CHARACTER VOICE PROFILES (write dialogue to match these styles):
{chr(10).join(char_voices)}
        """

    # Build adjacent page context for scene continuity
    continuity_context = ""
    continuity_parts = []

    # Use ACTUAL previous page script if available (for dialogue flow)
    if prev_page_script:
        # Extract the last 2-3 panels' dialogue from the previous page
        prev_panels = prev_page_script.get('panels', [])[-3:]
        prev_dialogue_lines = []
        for panel in prev_panels:
            # Handle both single dialogue and multiple bubbles
            bubbles = panel.get('dialogue_bubbles', [])
            if bubbles:
                for bubble in bubbles:
                    speaker = bubble.get('speaker', 'Unknown')
                    text = bubble.get('text', '')
                    if text:
                        prev_dialogue_lines.append(f'    {speaker}: "{text}"')
            elif panel.get('dialogue'):
                chars = panel.get('characters', ['Someone'])
                prev_dialogue_lines.append(f'    {chars[0] if chars else "Someone"}: "{panel["dialogue"]}"')
            if panel.get('caption'):
                prev_dialogue_lines.append(f'    [Caption]: "{panel["caption"]}"')

        if prev_dialogue_lines:
            continuity_parts.append(f"""PREVIOUS PAGE DIALOGUE (Page {prev_page_script.get('page_number')}) - CONTINUE THIS CONVERSATION:
{chr(10).join(prev_dialogue_lines)}

Your dialogue MUST flow naturally from this. Don't repeat what was said. Continue or respond to it.""")

    # Also include blueprint context for scene setup
    if prev_page:
        prev_chars = ', '.join(prev_page.get('key_characters', []))
        prev_scene = prev_page.get('scene_type', 'dialogue')
        continuity_parts.append(f"PREVIOUS PAGE CONTEXT ({prev_page.get('page_number')}): {prev_page.get('summary', '')} (Characters: {prev_chars}, Scene: {prev_scene})")

    if next_page:
        next_chars = ', '.join(next_page.get('key_characters', []))
        next_scene = next_page.get('scene_type', 'dialogue')
        continuity_parts.append(f"NEXT PAGE PREVIEW ({next_page.get('page_number')}): {next_page.get('summary', '')} (Characters: {next_chars}, Scene: {next_scene})")

    if continuity_parts:
        continuity_context = f"""
        SCENE CONTINUITY (CRITICAL for dialogue flow):
        {chr(10).join(continuity_parts)}

        - Continue conversations naturally - don't restart topics already discussed
        - If characters appear on previous page, maintain their emotional state
        - Set up any reveals/events that happen on the next page
        """

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
        {voice_context}
        {continuity_context}
        BLUEPRINT FOR THIS PAGE:
        Summary: {blueprint_item['summary']}
        Mood: {blueprint_item['mood']}
        Visual Notes: {blueprint_item.get('visual_notes', '')}
        Scene Type: {scene_type}
        Suggested Panel Count: {suggested_panel_count}
        Recommended Splash Panel: {recommended_splash}
        Is Full-Bleed Page: {is_spread}
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
        4. DIALOGUE LENGTH - Match text length to panel_size for visual balance:
           - 'large' panels (50%+ of page): 80-120 characters per dialogue bubble
           - 'medium' panels (25-50%): 50-80 characters per dialogue bubble
           - 'small' panels (15-25%): 20-50 characters per dialogue bubble
           Choose your words carefully to fit the space while advancing the story.
           Shorter is often better - let the art tell the story.

        5. For character DIALOGUE, you have TWO options:
           OPTION A - Single speaker: Use 'dialogue' field
           OPTION B - Multiple speakers: Use 'dialogue_bubbles' array for 2-3 speakers in same panel:
             [{{"speaker": "Character Name", "text": "Their line", "position": "top-left"}}]
             Positions: "top-left", "top-right", "bottom-left", "bottom-right"
           - Keep dialogue punchy and natural - no stage directions
           - If dialogue is interrupted, use em-dash: "Wait—"

        6. CAPTION LENGTH - Match narration to panel_size:
           - 'large' panels: 100-150 characters
           - 'medium' panels: 60-100 characters
           - 'small' panels: 30-60 characters

        7. If a panel has NO dialogue or caption, OMIT those fields entirely (do NOT use "null" or empty strings).
        8. For EACH panel, provide a STRUCTURED 'advice' object with:
           - 'scene_type': "{scene_type}"
           - 'required_gear': Object mapping character names to gear list
           - 'era_constraints': Era-specific requirements
           - 'continuity': {{'from_previous': "...", 'to_next': "..."}}
           - 'composition': {{'negative_space': "top-left|top-right|bottom-left|bottom-right"}}

        PACING RULES:
        {'- This page should have ONE dominant splash panel (large) with supporting smaller panels.' if recommended_splash else ''}
        {'- This is a FULL-BLEED PAGE - a single dramatic image covering the entire page. Make it EPIC and cinematic!' if is_spread else ''}
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
                response_schema=PAGE_SCRIPT_SCHEMA
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

        # Preserve blueprint layout metadata for downstream agents (LayoutAgent, Compositor)
        result['recommended_splash'] = recommended_splash
        result['is_full_bleed'] = is_spread  # Renamed: "spread" → "full_bleed" for digital (single page)
        result['is_cliffhanger'] = is_cliffhanger
        result['suggested_panel_count'] = suggested_panel_count
        result['scene_type'] = scene_type

        # Post-process: Remove "null" strings and clean punctuation (no truncation)
        def clean_text(text: str) -> str:
            """Clean punctuation patterns. Trust LLM to follow length guidelines."""
            if not text:
                return text
            # Clean up bad punctuation patterns
            text = re.sub(r'\.\.\.!', '—', text)  # ...! -> em-dash
            text = re.sub(r'\?\!', '?', text)  # ?! -> ?
            text = re.sub(r'\!\?', '!', text)  # !? -> !
            text = re.sub(r'\.\.\.\.+', '...', text)  # Multiple dots -> three
            text = re.sub(r'!!+', '!', text)  # Multiple ! -> one
            text = re.sub(r'\?\?+', '?', text)  # Multiple ? -> one
            return text

        def is_null_or_empty(text) -> bool:
            """Check if text is null, "null", None, or empty."""
            if text is None:
                return True
            if isinstance(text, str):
                return text.strip().lower() in ("", "null", "none")
            return False

        for panel in result.get('panels', []):
            # Remove "null" strings from dialogue field
            if is_null_or_empty(panel.get('dialogue')):
                panel.pop('dialogue', None)
            elif panel.get('dialogue'):
                panel['dialogue'] = clean_text(panel['dialogue'])

            # Clean dialogue_bubbles array - remove null/empty entries
            bubbles = panel.get('dialogue_bubbles', [])
            if bubbles:
                cleaned_bubbles = []
                for bubble in bubbles:
                    if not is_null_or_empty(bubble.get('text')):
                        bubble['text'] = clean_text(bubble['text'])
                        cleaned_bubbles.append(bubble)
                if cleaned_bubbles:
                    panel['dialogue_bubbles'] = cleaned_bubbles
                else:
                    panel.pop('dialogue_bubbles', None)

            # Remove "null" strings from caption field
            if is_null_or_empty(panel.get('caption')):
                panel.pop('caption', None)
            elif panel.get('caption'):
                panel['caption'] = clean_text(panel['caption'])

        return result
