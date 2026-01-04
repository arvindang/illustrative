import asyncio
import os
import json
from pathlib import Path
from scripting_agent import ScriptingAgent
from character_architect import CharacterArchitect
from illustrator_agent import IllustratorAgent
from compositor_agent import CompositorAgent

async def run_smoke_test():
    print("🚀 Starting Micro-Book Smoke Test...")
    
    input_file = "assets/input/20-thousand-leagues-under-the-sea.txt"
    style = "Manga/Anime"  # Try: "Ligne Claire (Franco-Belgian)", "Expressionist", "Sketch/Pencil Drawing", etc.
    tone = "Philosophical"  # Try: "Satirical", "Tragic", "Noir Detective", etc.
    
    # 1. Scripting (Test Mode = 1 page)
    print("\n--- STEP 1: SCRIPTING ---")
    scripter = ScriptingAgent(input_file)
    script = await scripter.generate_script(style=style, test_mode=True)
    
    script_path = "assets/output/20-thousand-leagues-under-the-sea_test_page.json"
    
    # 2. Character Design
    print("\n--- STEP 2: CHARACTER DESIGN ---")
    architect = CharacterArchitect(script_path)
    await architect.design_all_assets(style=style)
        
    # 3. Illustration (Checkpointing will be active)
    print("\n--- STEP 3: ILLUSTRATION ---")
    illustrator = IllustratorAgent(script_path, f"{style} style, {tone} tone")
    await illustrator.run_production()
    
    # 4. Composition
    print("\n--- STEP 4: COMPOSITION ---")
    optimized_script = Path("assets/output/20-thousand-leagues-under-the-sea_test_page_optimized.json")
    final_script = str(optimized_script) if optimized_script.exists() else script_path
    compositor = CompositorAgent(final_script)
    compositor.run()
    
    print("\n✅ Smoke Test Complete! Check assets/output/final_pages/page_1.png")

if __name__ == "__main__":
    asyncio.run(run_smoke_test())
