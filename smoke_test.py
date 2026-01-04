import asyncio
from pathlib import Path
from scripting_agent import ScriptingAgent
from illustrator_agent import IllustratorAgent
from compositor_agent import CompositorAgent

async def run_smoke_test():
    """
    Simplified 3-step smoke test:
    1. ScriptingAgent: Script + asset manifest
    2. IllustratorAgent: Reference sheets + panels
    3. CompositorAgent: Compose + export
    """
    print("🚀 Starting Smoke Test (Simplified 3-Step Pipeline)...")

    input_file = "assets/input/20-thousand-leagues-under-the-sea.txt"
    style = "Manga/Anime"
    tone = "Philosophical"

    script_path = "assets/output/20-thousand-leagues-under-the-sea_test_page.json"

    # --- STEP 1: SCRIPTING (includes asset manifest) ---
    print("\n--- STEP 1: SCRIPTING + ASSET ANALYSIS ---")
    scripter = ScriptingAgent(input_file)
    script = await scripter.generate_script(style=style, test_mode=True)

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

    print("\n✅ Smoke Test Complete! Check assets/output/final_pages/")

if __name__ == "__main__":
    asyncio.run(run_smoke_test())
