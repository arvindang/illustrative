import asyncio
import os
import json
from pathlib import Path
from scripting_agent import ScriptingAgent
from character_architect import CharacterArchitect
from illustrator_agent import IllustratorAgent
from compositor_agent import CompositorAgent

async def run_production():
    print("🚀 Starting 100-Page Production Run...")
    
    input_file = "assets/input/20-thousand-leagues-under-the-sea.txt"
    style = "Botanical Illustration"
    tone = "Melancholic"
    
    # 1. Scripting (Production Mode = 100 pages)
    print("\n--- STEP 1: SCRIPTING ---")
    scripter = ScriptingAgent(input_file)
    
    # We need to analyze global context first to match app.py behavior
    full_text = scripter.load_content(test_mode=False)
    context_constraints = await scripter.analyze_global_context(full_text)
    print(f"📌 Context Detected: {context_constraints}")

    script = await scripter.generate_script(
        style=style, 
        tone=tone, 
        writing_style="Cinematic", 
        test_mode=False,
        context_constraints=context_constraints
    )
    
    script_path = "assets/output/20-thousand-leagues-under-the-sea_full_script.json"
    
    # 2. Character Design
    print("\n--- STEP 2: CHARACTER DESIGN ---")
    architect = CharacterArchitect(script_path)
    await architect.design_all_characters(style=style)
        
    # 3. Illustration (Resumes from manifest)
    print("\n--- STEP 3: ILLUSTRATION ---")
    illustrator = IllustratorAgent(script_path, f"{style} style, {tone} tone")
    await illustrator.run_production()
    
    # 4. Composition
    print("\n--- STEP 4: COMPOSITION ---")
    optimized_script = Path("assets/output/20-thousand-leagues-under-the-sea_full_script_optimized.json")
    final_script = str(optimized_script) if optimized_script.exists() else script_path
    compositor = CompositorAgent(final_script)
    compositor.run()
    
    print("\n✅ Production Complete! Check assets/output/final_pages/ for 100 pages.")

if __name__ == "__main__":
    asyncio.run(run_production())
