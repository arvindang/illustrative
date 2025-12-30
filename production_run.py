import asyncio
import os
import json
from pathlib import Path
from scripting_agent import ScriptingAgent
from character_architect import CharacterArchitect
from illustrator_agent import IllustratorAgent
from compositor_agent import CompositorAgent

async def run_production():
    print("🚀 Starting Production Run...")

    input_file = "assets/input/20-thousand-leagues-under-the-sea.txt"
    style = "Art Nouveau"  # Try: "Manga/Anime", "Chiaroscuro", "Vintage Pulp", etc.
    tone = "Romantic"  # Try: "Philosophical", "Cosmic Horror", "Gothic", etc.

    # Auto-calculate page count
    scripter = ScriptingAgent(input_file)
    full_text = scripter.load_content(test_mode=False)

    from utils import calculate_page_count
    word_count = len(full_text.split())

    # Set to None for auto, or specify number to override (e.g., 75)
    user_override = None  # CHANGE THIS to set specific page count

    page_calc = calculate_page_count(
        word_count=word_count,
        test_mode=False,
        user_override=user_override
    )

    print(f"\n📊 PROJECT SCOPE")
    print(f"   Input: {word_count:,} words")
    print(f"   Category: {page_calc['density_category']}")
    print(f"   Target: {page_calc['recommended']} pages")
    print(f"   Range: {page_calc['minimum']}-{page_calc['maximum']} pages")
    print(f"   Est. time: ~{page_calc['estimated_time_minutes']} min (~{page_calc['estimated_time_minutes']//60}h {page_calc['estimated_time_minutes']%60}m)")
    if page_calc['warning']:
        print(f"   ⚠️  {page_calc['warning']}")

    # Confirmation for large projects
    if page_calc['recommended'] >= 100:
        confirm = input(f"\n⏰ This will take ~{page_calc['estimated_time_minutes']//60}h {page_calc['estimated_time_minutes']%60}m. Proceed? (y/n): ")
        if confirm.lower() != 'y':
            print("❌ Production cancelled.")
            return
    print()

    # 1. Scripting
    print("\n--- STEP 1: SCRIPTING ---")
    context_constraints = await scripter.analyze_global_context(full_text)
    print(f"📌 Context: {context_constraints}")

    script = await scripter.generate_script(
        style=style,
        tone=tone,
        writing_style="Cinematic",
        test_mode=False,
        context_constraints=context_constraints,
        target_page_override=page_calc['recommended']
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
    
    print(f"\n✅ Production Complete! Generated {page_calc['recommended']} pages. Check assets/output/final_pages/")

if __name__ == "__main__":
    asyncio.run(run_production())
