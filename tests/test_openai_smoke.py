"""
Smoke test for the OpenAI image backend (reference sheets + panels + composition).

Skips the Gemini scripting step by using a pre-built minimal script JSON.
Tests: OpenAI client → Reference generation → Panel generation → Composition.

Usage:
    # Run with venv
    python tests/test_openai_smoke.py

    # Or via pytest
    pytest tests/test_openai_smoke.py -v -s
"""
import asyncio
import json
import sys
import uuid
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import config

# ---- Minimal test data (no Gemini required) ----

MINI_SCRIPT = [
    {
        "page_number": 1,
        "panels": [
            {
                "panel_id": 1,
                "visual_description": (
                    "A vast underwater scene viewed through the large porthole of "
                    "the Nautilus. Captain Nemo stands silhouetted against the glass, "
                    "arms clasped behind his back, gazing at a luminous coral reef "
                    "teeming with fish. The interior is dimly lit by brass oil lamps."
                ),
                "characters": ["Captain Nemo"],
                "dialogue": "The sea is everything.",
                "caption": "Beneath the waves, a world unknown to mankind.",
                "bubble_position": "top-left",
                "shot_type": "establishing",
                "panel_size": "large",
                "key_objects": ["The Nautilus"],
                "advice": {
                    "scene_type": "establishing",
                    "continuity_notes": "Opening scene",
                    "historical_constraints": "1860s Victorian submarine",
                    "character_gear": "Captain Nemo wears naval uniform",
                    "composition": {"negative_space": "top-left"},
                },
            },
            {
                "panel_id": 2,
                "visual_description": (
                    "Close-up of Professor Aronnax writing in his leather-bound "
                    "journal by candlelight. His spectacles rest on the bridge of "
                    "his nose. A detailed anatomical sketch of a narwhal is visible "
                    "on the open page."
                ),
                "characters": ["Professor Aronnax"],
                "dialogue": "",
                "caption": "I recorded every wonder, lest the world think me mad.",
                "bubble_position": "bottom-right",
                "shot_type": "close-up",
                "panel_size": "medium",
                "key_objects": [],
                "advice": {
                    "scene_type": "introspection",
                    "continuity_notes": "Follows establishing shot",
                    "historical_constraints": "1860s writing implements",
                    "character_gear": "Aronnax wears tweed jacket, spectacles",
                    "composition": {"negative_space": "bottom-right"},
                },
            },
            {
                "panel_id": 3,
                "visual_description": (
                    "Medium two-shot of Captain Nemo and Professor Aronnax standing "
                    "at the helm of the Nautilus. Nemo gestures toward a large "
                    "nautical map pinned to the wall. Aronnax looks astonished."
                ),
                "characters": ["Captain Nemo", "Professor Aronnax"],
                "dialogue": "We shall journey to places no surface-dweller has seen.",
                "caption": "",
                "bubble_position": "top-right",
                "shot_type": "two-shot",
                "panel_size": "medium",
                "key_objects": ["The Nautilus"],
                "advice": {
                    "scene_type": "dialogue",
                    "continuity_notes": "First meeting at the helm",
                    "historical_constraints": "Victorian naval instruments",
                    "character_gear": "Both in period attire",
                    "composition": {"negative_space": "top-right"},
                },
            },
        ],
    }
]

MINI_ASSETS = {
    "characters": [
        {
            "name": "Captain Nemo",
            "aliases": ["Nemo", "The Captain"],
            "description": (
                "A tall, enigmatic man in his 40s with dark piercing eyes and a "
                "neatly trimmed beard. Wears an ornate navy-blue naval uniform with "
                "gold epaulettes and brass buttons. Commands absolute authority."
            ),
            "age_range": "40s",
            "occupation": "Captain of the Nautilus",
            "distinctive_items": ["ornate naval uniform", "brass telescope"],
            "color_signature": "#1B3A5C",
        },
        {
            "name": "Professor Aronnax",
            "aliases": ["Aronnax", "Pierre Aronnax", "The Professor"],
            "description": (
                "A scholarly French naturalist in his 50s with grey-streaked hair, "
                "round wire-rimmed spectacles, and a kind but curious expression. "
                "Wears a brown tweed jacket over a white shirt with a cravat."
            ),
            "age_range": "50s",
            "occupation": "Marine Biologist / Professor",
            "distinctive_items": ["wire-rimmed spectacles", "leather journal"],
            "color_signature": "#8B6914",
        },
    ],
    "objects": [
        {
            "name": "The Nautilus",
            "description": (
                "A magnificent Victorian-era submarine shaped like an elongated "
                "spindle. Riveted iron hull with large circular portholes emitting "
                "a warm amber glow. The interior is lavishly decorated with "
                "mahogany panels, velvet drapes, and brass instruments."
            ),
            "key_features": [
                "riveted iron hull",
                "large circular portholes",
                "ornate Victorian interior",
            ],
            "condition": "pristine, well-maintained",
            "material_context": "1860s industrial iron and brass",
        }
    ],
    "color_script": {
        "act_1": {"palette": ["deep ocean blue", "amber lamplight", "brass gold"]},
    },
}


async def run_openai_smoke_test():
    """Run OpenAI image pipeline smoke test."""
    print("=" * 60)
    print("OPENAI IMAGE BACKEND SMOKE TEST")
    print("=" * 60)
    print(f"  Backend:       {config.image_backend}")
    print(f"  Batch enabled: {config.openai_batch_enabled}")
    print(f"  Primary model: {config.openai_image_model_primary}")
    print(f"  Panel size:    {config.openai_panel_size}")
    print(f"  Ref quality:   {config.openai_ref_image_quality}")
    print(f"  API key set:   {'Yes' if config.openai_api_key else 'NO'}")
    print("=" * 60)

    if not config.openai_api_key:
        print("\nERROR: OPENAI_API_KEY not set. Add it to .env and retry.")
        return

    # Create isolated output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    short_id = str(uuid.uuid4())[:8]
    base_dir = Path(f"assets/output/openai_smoke_{timestamp}_{short_id}")
    base_dir.mkdir(parents=True, exist_ok=True)
    (base_dir / "characters").mkdir(exist_ok=True)
    (base_dir / "objects").mkdir(exist_ok=True)
    (base_dir / "pages").mkdir(exist_ok=True)
    (base_dir / "final_pages").mkdir(exist_ok=True)

    # Write mini script + assets
    stem = "20K_Leagues_Under_the_Sea"
    script_path = base_dir / f"{stem}_test_page.json"
    assets_path = base_dir / f"{stem}_assets.json"

    with open(script_path, "w") as f:
        json.dump(MINI_SCRIPT, f, indent=2)
    with open(assets_path, "w") as f:
        json.dump(MINI_ASSETS, f, indent=2)

    print(f"\n  Output dir: {base_dir}")
    print(f"  Script:     {script_path}")
    print(f"  Assets:     {assets_path}")

    # --- STEP 1: Get OpenAI image agents via factory ---
    print("\n--- STEP 1: Initialize OpenAI image agents ---")
    from agents import get_image_agents

    style_prompt = "Lush Watercolor style, Philosophical tone"
    ref_agent, panel_agent = get_image_agents(
        str(script_path), style_prompt, base_output_dir=base_dir
    )
    print(f"  Reference agent: {type(ref_agent).__name__}")
    print(f"  Panel agent:     {type(panel_agent).__name__}")

    # --- STEP 2: Generate reference sheets ---
    print("\n--- STEP 2: Generate reference sheets (characters + objects) ---")
    await ref_agent.generate_all_references(style="Lush Watercolor")

    # Verify references
    char_dir = base_dir / "characters"
    char_count = len([d for d in char_dir.iterdir() if d.is_dir() and (d / "metadata.json").exists()])
    obj_dir = base_dir / "objects"
    obj_count = len([d for d in obj_dir.iterdir() if d.is_dir() and (d / "metadata.json").exists()])
    print(f"  Characters designed: {char_count}")
    print(f"  Objects designed:    {obj_count}")

    assert char_count > 0, "No character references generated!"

    # --- STEP 3: Generate panel images ---
    print("\n--- STEP 3: Generate panel images ---")
    await panel_agent.run_production()

    # Verify panels
    pages_dir = base_dir / "pages"
    panel_count = len(list(pages_dir.rglob("*.png")))
    print(f"  Panel images generated: {panel_count}")

    assert panel_count > 0, "No panel images generated!"

    # --- STEP 4: Composition ---
    print("\n--- STEP 4: Composition + export ---")
    from agents import CompositorAgent

    compositor = CompositorAgent(str(script_path), base_output_dir=base_dir)
    compositor.run()

    final_dir = base_dir / "final_pages"
    final_count = len(list(final_dir.glob("*.png")))
    print(f"  Final pages composed: {final_count}")

    # --- RESULTS ---
    print("\n" + "=" * 60)
    print("SMOKE TEST RESULTS")
    print("=" * 60)
    print(f"  Characters:    {char_count}")
    print(f"  Objects:       {obj_count}")
    print(f"  Panels:        {panel_count}")
    print(f"  Final pages:   {final_count}")
    print(f"  Output:        {base_dir}")

    if final_count > 0:
        print("\n  PASSED - OpenAI image backend is working!")
    else:
        print("\n  PARTIAL - Images generated but composition may need debugging")

    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(run_openai_smoke_test())
