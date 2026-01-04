import asyncio
import os
import json
from pathlib import Path
from scripting_agent import ScriptingAgent
from character_architect import CharacterArchitect
from illustrator_agent import IllustratorAgent
from compositor_agent import CompositorAgent

async def run_production():
    print("🚀 Starting Production Run (Director Mode)...")

    input_file = "assets/input/20-thousand-leagues-under-the-sea.txt"
    style = "Lush Watercolor"
    tone = "Philosophical"
    
    # Simplified Scope: The Director (Gemini) handles the pacing for this target. 
    target_pages = 10 

    print(f"   Input: {input_file}")
    print(f"   Target: {target_pages} pages")
    print(f"   Style: {style}")
    print()

    # 1. Scripting
    print("\n--- STEP 1: SCRIPTING ---")
    scripter = ScriptingAgent(input_file)
    
    # The 'Director' agent (ScriptingAgent) now handles the full adaptation 
    # using Gemini's Context Caching and large context window.
    script = await scripter.generate_script(
        style=f"{style}, {tone}",
        test_mode=False,
        target_page_override=target_pages
    )
    
    # The output path is determined by ScriptingAgent, but we know standard location
    script_path = "assets/output/20-thousand-leagues-under-the-sea_full_script.json"
    
    # 2. Character Design
    print("\n--- STEP 2: ASSET DESIGN (Characters & Objects) ---")
    architect = CharacterArchitect(script_path)
    await architect.design_all_assets(style=style)

    # 2.5. Continuity Validation
    print("\n--- STEP 2.5: CONTINUITY VALIDATION ---")
    try:
        from continuity_validator import ContinuityValidator
        validator = ContinuityValidator(script_path, "assets/output/characters")
        issues = validator.validate_script()

        if issues['errors']:
            print(f"⚠️  Found {len(issues['errors'])} continuity errors (Check logs).")
        else:
            print("✅ Continuity validation passed")
    except Exception as e:
        print(f"⚠️  Continuity validation skipped: {e}")

    # 3. Illustration (Resumes from manifest)
    print("\n--- STEP 3: ILLUSTRATION ---")
    illustrator = IllustratorAgent(script_path, f"{style} style, {tone} tone")
    await illustrator.run_production()
    
    # 4. Composition
    print("\n--- STEP 4: COMPOSITION ---")
    compositor = CompositorAgent(script_path)
    compositor.run()
    
    print(f"\n✅ Production Complete! Generated {target_pages} pages.")
    print(f"   Check: assets/output/final_pages/")

if __name__ == "__main__":
    asyncio.run(run_production())