"""
PASS 5.5: Dialogue Polish

Reads the full generated script and improves dialogue flow:
- Fixes abrupt topic changes between pages
- Removes repeated phrases/words across panels
- Ensures consistent character voice
- Smooths transitions between scenes
"""
from google.genai import types

from utils import get_tpm_limiter, estimate_tokens_for_text, get_client
from config import config
from agents.scripting.schemas import DIALOGUE_POLISH_SCHEMA
from agents.scripting.utils import get_scribe_limiter


def _window_needs_polish(pages_in_window):
    """Quick check if this window's dialogue likely needs polishing."""
    all_dialogues = []
    for page in pages_in_window:
        for panel in page.get('panels', []):
            d = panel.get('dialogue', '')
            if d:
                all_dialogues.append(d.lower())
            for bubble in panel.get('dialogue_bubbles', []):
                t = bubble.get('text', '')
                if t:
                    all_dialogues.append(t.lower())

    if not all_dialogues:
        return False  # No dialogue to polish

    # Check for repeated phrases (same dialogue appearing 2+ times)
    if len(all_dialogues) != len(set(all_dialogues)):
        return True

    # If all dialogue is short and varied, likely clean
    return True  # Default to polishing (conservative)


async def polish_dialogue(
    full_script: list,
    character_arcs: dict,
    context_constraints: str = ""
) -> list:
    """
    PASS 5.5: DIALOGUE POLISH PASS

    Reads the full generated script and improves dialogue flow:
    - Fixes abrupt topic changes between pages
    - Removes repeated phrases/words across panels
    - Ensures consistent character voice
    - Smooths transitions between scenes

    This pass processes pages in batches to maintain context while
    staying within token limits.

    Args:
        full_script: The complete script from Pass 5
        character_arcs: Character arc data with voice profiles
        context_constraints: Era/setting constraints

    Returns:
        Polished script with improved dialogue
    """
    if not full_script:
        return full_script

    print(f"\n💬 PASS 5.5: Polishing dialogue across {len(full_script)} pages...")

    # Build character voice reference for the LLM
    voice_reference = ""
    if character_arcs:
        voices = []
        for char in character_arcs.get('characters', [])[:8]:  # Top 8 characters
            name = char.get('name', '')
            voice = char.get('voice_profile', {})
            samples = char.get('dialogue_samples', [])
            if name and (voice or samples):
                voice_info = f"  {name}:"
                if voice.get('formality'):
                    voice_info += f" {voice.get('formality')} tone,"
                if voice.get('vocabulary_style'):
                    voice_info += f" {voice.get('vocabulary_style')},"
                if voice.get('dialect_markers'):
                    voice_info += f" uses: {', '.join(voice.get('dialect_markers', [])[:3])}"
                if samples:
                    voice_info += f"\n    Example: \"{samples[0][:80]}...\""
                voices.append(voice_info)

        if voices:
            voice_reference = f"""
CHARACTER VOICE PROFILES (dialogue must match these):
{chr(10).join(voices)}
"""

    # Process in windows of 5 pages (overlapping by 1 for context)
    WINDOW_SIZE = 5
    polished_script = []

    for i in range(0, len(full_script), WINDOW_SIZE - 1):
        window = full_script[i:i + WINDOW_SIZE]
        window_start = window[0]['page_number']
        window_end = window[-1]['page_number']

        # Skip if already processed (from overlap)
        if polished_script and window[0]['page_number'] <= polished_script[-1]['page_number']:
            window = window[1:]  # Skip first page (already polished)
            if not window:
                continue

        # Quick heuristic: skip windows with no dialogue
        if not _window_needs_polish(window):
            print(f"   ⏭️  Skipping pages {window_start}-{window_end} (no dialogue to polish)")
            for page in window:
                if not polished_script or page['page_number'] > polished_script[-1]['page_number']:
                    polished_script.append(page)
            continue

        print(f"   📝 Polishing pages {window_start}-{window_end}...")

        # Build dialogue summary for each page in window
        window_dialogue = []
        for page in window:
            page_lines = [f"PAGE {page['page_number']}:"]
            for panel in page.get('panels', []):
                panel_id = panel.get('panel_id', '?')
                chars = panel.get('characters', [])

                # Collect all dialogue from this panel
                dialogues = []
                if panel.get('dialogue'):
                    speaker = chars[0] if chars else 'Unknown'
                    dialogues.append(f'{speaker}: "{panel["dialogue"]}"')

                for bubble in panel.get('dialogue_bubbles', []):
                    speaker = bubble.get('speaker', 'Unknown')
                    text = bubble.get('text', '')
                    if text:
                        dialogues.append(f'{speaker}: "{text}"')

                if panel.get('caption'):
                    dialogues.append(f'[CAPTION]: "{panel["caption"]}"')

                if dialogues:
                    page_lines.append(f"  Panel {panel_id}: {' / '.join(dialogues)}")

            window_dialogue.append('\n'.join(page_lines))

        # Build polish prompt
        prompt = f"""
You are a Graphic Novel Dialogue Editor. Your job is to polish dialogue for flow and consistency.

{voice_reference}

CURRENT DIALOGUE (pages {window_start}-{window_end}):
{chr(10).join(window_dialogue)}

{f'ERA CONSTRAINTS: {context_constraints}' if context_constraints else ''}

IDENTIFY AND FIX:
1. ABRUPT TOPIC CHANGES: If characters suddenly change topic without transition, add a bridging line or smooth the shift
2. REPEATED PHRASES: If the same phrase/word appears multiple times across panels, vary it
3. VOICE INCONSISTENCY: If a character speaks out of character (too formal/informal for their profile), fix it
4. STILTED DIALOGUE: If dialogue sounds unnatural, make it conversational while preserving meaning
5. INTERRUPTED FLOW: If a conversation on one page doesn't connect to the next, add continuity

RULES:
- Keep each dialogue line under 120 characters
- Keep each caption under 140 characters
- Preserve the MEANING of each line - only improve the WORDING
- If dialogue is already good, return it unchanged
- Maintain consistent character voices throughout

OUTPUT FORMAT: Return a JSON array of objects, one per page:
[
  {{
    "page_number": N,
    "panels": [
      {{
        "panel_id": N,
        "dialogue": "polished dialogue or null if none",
        "dialogue_bubbles": [
          {{"speaker": "Name", "text": "polished text", "position": "top-left"}}
        ],
        "caption": "polished caption or null if none",
        "changes_made": "brief description of changes or 'none'"
      }}
    ]
  }}
]
"""

        async with get_scribe_limiter():
            await get_tpm_limiter().acquire(estimate_tokens_for_text(prompt))

            try:
                response = await get_client().aio.models.generate_content(
                    model=config.scripting_model_page_script,
                    contents=[prompt],
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=DIALOGUE_POLISH_SCHEMA
                    )
                )

                polish_result = response.parsed

                # Merge polished dialogue back into original script
                for polished_page in polish_result:
                    page_num = polished_page['page_number']
                    # Find original page in window
                    original = next((p for p in window if p['page_number'] == page_num), None)
                    if not original:
                        continue

                    # Track changes for logging
                    changes_count = 0

                    for polished_panel in polished_page.get('panels', []):
                        panel_id = polished_panel['panel_id']
                        # Find original panel
                        orig_panel = next(
                            (p for p in original.get('panels', []) if p.get('panel_id') == panel_id),
                            None
                        )
                        if not orig_panel:
                            continue

                        # Apply polished dialogue if provided
                        if polished_panel.get('dialogue'):
                            if orig_panel.get('dialogue') != polished_panel['dialogue']:
                                orig_panel['dialogue'] = polished_panel['dialogue']
                                changes_count += 1

                        # Apply polished bubbles if provided
                        if polished_panel.get('dialogue_bubbles'):
                            orig_panel['dialogue_bubbles'] = polished_panel['dialogue_bubbles']
                            changes_count += 1

                        # Apply polished caption if provided
                        if polished_panel.get('caption'):
                            if orig_panel.get('caption') != polished_panel['caption']:
                                orig_panel['caption'] = polished_panel['caption']
                                changes_count += 1

                    polished_script.append(original)
                    if changes_count > 0:
                        print(f"      Page {page_num}: {changes_count} dialogue improvements")

            except Exception as e:
                print(f"   ⚠️  Polish failed for pages {window_start}-{window_end}: {e}")
                # On failure, keep original dialogue
                for page in window:
                    if not polished_script or page['page_number'] > polished_script[-1]['page_number']:
                        polished_script.append(page)

    # Ensure all pages are included (in case of edge cases)
    polished_page_nums = {p['page_number'] for p in polished_script}
    for page in full_script:
        if page['page_number'] not in polished_page_nums:
            polished_script.append(page)

    polished_script.sort(key=lambda x: x['page_number'])
    print(f"   ✅ Dialogue polish complete")

    return polished_script
