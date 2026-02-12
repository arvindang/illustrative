import asyncio
from pathlib import Path
from agents import ScriptingAgent, CompositorAgent, get_image_agents
from config import config

async def run_production():
    """
    Simplified 3-step production pipeline:
    1. ScriptingAgent: Generates script + asset manifest (characters & objects descriptions)
    2. Image Agents: Generates reference sheets + panel images (Gemini or OpenAI backend)
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
    print(f"   Image backend: {config.image_backend}")
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
    style_prompt = f"{style} style, {tone} tone"
    ref_agent, panel_agent = get_image_agents(script_path, style_prompt)

    # Generate reference sheets for characters and objects
    await ref_agent.generate_all_references(style=style)

    # Generate panel images
    await panel_agent.run_production()

    # --- STEP 3: COMPOSITION + EXPORT ---
    print("\n--- STEP 3: COMPOSITION + EXPORT ---")
    compositor = CompositorAgent(script_path)
    compositor.run()

    print(f"\n✅ Production Complete! Generated {target_pages} pages.")
    print(f"   Check: assets/output/final_pages/")

if __name__ == "__main__":
    asyncio.run(run_production())