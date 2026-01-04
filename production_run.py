import asyncio
from pathlib import Path
from scripting_agent import ScriptingAgent
from illustrator_agent import IllustratorAgent
from compositor_agent import CompositorAgent

async def run_production():
    """
    Simplified 3-step production pipeline:
    1. ScriptingAgent: Generates script + asset manifest (characters & objects descriptions)
    2. IllustratorAgent: Generates reference sheets + panel images
    3. CompositorAgent: Composes pages + exports PDF/EPUB
    """
    print("🚀 Starting Production Run (Simplified Pipeline)...")

    input_file = "assets/input/20-thousand-leagues-under-the-sea.txt"
    style = "Lush Watercolor"
    tone = "Philosophical"
    target_pages = 10

    print(f"   Input: {input_file}")
    print(f"   Target: {target_pages} pages")
    print(f"   Style: {style}")
    print()

    # Derive script path
    input_stem = Path(input_file).stem
    script_path = f"assets/output/{input_stem}_full_script.json"

    # --- STEP 1: SCRIPTING (includes asset manifest generation) ---
    print("\n--- STEP 1: SCRIPTING + ASSET ANALYSIS ---")
    scripter = ScriptingAgent(input_file)
    script = await scripter.generate_script(
        style=f"{style}, {tone}",
        test_mode=False,
        target_page_override=target_pages
    )

    # --- STEP 2: ILLUSTRATION (reference sheets + panels) ---
    print("\n--- STEP 2: ILLUSTRATION ---")
    illustrator = IllustratorAgent(script_path, f"{style} style, {tone} tone")

    # Generate reference sheets for characters and objects
    await illustrator.generate_all_references(style=style)

    # Generate panel images
    await illustrator.run_production()

    # --- STEP 3: COMPOSITION + EXPORT ---
    print("\n--- STEP 3: COMPOSITION + EXPORT ---")
    compositor = CompositorAgent(script_path)
    compositor.run()

    print(f"\n✅ Production Complete! Generated {target_pages} pages.")
    print(f"   Check: assets/output/final_pages/")

if __name__ == "__main__":
    asyncio.run(run_production())