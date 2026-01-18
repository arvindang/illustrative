"""
PASS 4: Asset Manifest

Generates visual descriptions for characters, objects, locations, and defines
interaction rules and color scripts for visual consistency.
"""
from typing import Callable

from google.genai import types

from utils import (
    retry_with_backoff, get_tpm_limiter,
    estimate_tokens_for_text, extract_token_usage,
    get_client
)
from config import config
from agents.scripting.schemas import ASSET_MANIFEST_SCHEMA


@retry_with_backoff()
async def generate_asset_manifest(
    cache_name: str,
    full_text_fallback: str,
    blueprint: list,
    style: str,
    context_constraints: str = "",
    character_arcs: dict = None,
    extract_characters_fn: Callable[[list], list] = None
) -> dict:
    """
    PASS 4: ASSET EXTRACTION (Enhanced)
    Generates visual descriptions for characters and key objects in a single pass.
    Now includes interaction_rules for scene-specific requirements.

    Args:
        cache_name: Gemini context cache reference
        full_text_fallback: Full text if cache unavailable
        blueprint: Page-by-page blueprint
        style: Art style for the graphic novel
        context_constraints: Era/setting constraints
        character_arcs: Character arc data (optional)
        extract_characters_fn: Function to extract characters from blueprint

    Returns:
        Asset manifest dict with characters, objects, locations, interaction_rules
    """
    # Extract unique characters from blueprint
    characters = extract_characters_fn(blueprint) if extract_characters_fn else []
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
            response_schema=ASSET_MANIFEST_SCHEMA
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
