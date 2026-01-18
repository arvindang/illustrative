"""
Quick smoke test for the complete pipeline.

This is a minimal test that runs all 3 steps with default settings
to verify the pipeline is working end-to-end.

Usage:
    # Run via pytest
    pytest tests/test_smoke.py -v

    # Run as standalone script
    python tests/test_smoke.py
"""
import asyncio
import sys
from pathlib import Path

import pytest

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents import ScriptingAgent, IllustratorAgent, CompositorAgent


@pytest.mark.asyncio
async def test_smoke():
    """
    Simplified 3-step smoke test:
    1. ScriptingAgent: Script + asset manifest
    2. IllustratorAgent: Reference sheets + panels
    3. CompositorAgent: Compose + export
    """
    print("\nStarting Smoke Test (Simplified 3-Step Pipeline)...")

    input_file = "assets/input/20-thousand-leagues-under-the-sea.txt"
    style = "Manga/Anime"
    tone = "Philosophical"

    script_path = "assets/output/20-thousand-leagues-under-the-sea_test_page.json"

    # --- STEP 1: SCRIPTING (includes asset manifest) ---
    print("\n--- STEP 1: SCRIPTING + ASSET ANALYSIS ---")
    scripter = ScriptingAgent(input_file)
    script = await scripter.generate_script(style=style, test_mode=True)

    assert script is not None, "Script generation failed"

    # --- STEP 2: ILLUSTRATION (reference sheets + panels) ---
    print("\n--- STEP 2: ILLUSTRATION ---")
    illustrator = IllustratorAgent(script_path, f"{style} style, {tone} tone")

    # Generate reference sheets for characters/objects
    await illustrator.generate_all_references(style=style)

    # Generate panel images
    await illustrator.run_production()

    # --- STEP 3: COMPOSITION + EXPORT ---
    print("\n--- STEP 3: COMPOSITION + EXPORT ---")
    compositor = CompositorAgent(script_path)
    compositor.run()

    print("\nSmoke Test Complete! Check assets/output/final_pages/")


# Allow running as standalone script
if __name__ == "__main__":
    asyncio.run(test_smoke())
