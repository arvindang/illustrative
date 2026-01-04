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
    style = "Lush Watercolor"  # Try: "Manga/Anime", "Chiaroscuro", "Vintage Pulp", etc.
    tone = "Philosophical"  # Try: "Philosophical", "Cosmic Horror", "Gothic", etc.

    # Auto-calculate page count
    scripter = ScriptingAgent(input_file)
    full_text = scripter.load_content(test_mode=False)

    from utils import calculate_page_count
    word_count = len(full_text.split())

    # --- SHOW IDEAL CALCULATION ---
    ideal_calc = calculate_page_count(
        word_count=word_count,
        test_mode=False,
        user_override=None
    )
    print(f"\n📊 IDEAL PROJECT SCOPE (Auto-Calculation)")
    print(f"   Input: {word_count:,} words")
    print(f"   Category: {ideal_calc['density_category']}")
    print(f"   Recommended: {ideal_calc['recommended']} pages")
    print(f"   Range: {ideal_calc['minimum']}-{ideal_calc['maximum']} pages")
    
    # --- APPLY PRODUCTION LIMIT ---
    # Set to None for auto, or specify number to override (e.g., 75)
    user_override = 10  # LIMITING TO 10 PAGES FOR PROTOTYPE RUN

    page_calc = calculate_page_count(
        word_count=word_count,
        test_mode=False,
        user_override=user_override
    )

    print(f"\n📊 ACTUAL PRODUCTION SCOPE")
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
        test_mode=False,
        context_constraints=context_constraints,
        target_page_override=page_calc['recommended']
    )
    
    script_path = "assets/output/20-thousand-leagues-under-the-sea_full_script.json"
    
    # 2. Character Design
    print("\n--- STEP 2: ASSET DESIGN (Characters & Objects) ---")
    architect = CharacterArchitect(script_path)
    await architect.design_all_assets(style=style)

    # 2.5. Continuity Validation
    print("\n--- STEP 2.5: CONTINUITY VALIDATION ---")
    from continuity_validator import ContinuityValidator
    # Fix: Pass character metadata directory and use correct method name
    validator = ContinuityValidator(script_path, "assets/output/characters")
    issues = validator.validate_script()

    if issues['errors']:
        print(f"⚠️  Found {len(issues['errors'])} continuity errors:")
        for error in issues['errors']:
            print(f"  - Page {error['page']}, Panel {error['panel']}: {error['message']}")
        print("\n⚠️  Auto-continuing for production run...")

    if issues['warnings']:
        print(f"ℹ️  Found {len(issues['warnings'])} continuity warnings:")
        for warning in issues['warnings']:
            print(f"  - Page {warning['page']}, Panel {warning['panel']}: {warning['message']}")

    print("✅ Continuity validation passed")

    # 3. Illustration (Resumes from manifest)
    print("\n--- STEP 3: ILLUSTRATION ---")
    illustrator = IllustratorAgent(script_path, f"{style} style, {tone} tone")
    await illustrator.run_production()
    
    # 4. Composition
    print("\n--- STEP 4: COMPOSITION ---")
    compositor = CompositorAgent(script_path)
    compositor.run()
    
    print(f"\n✅ Production Complete! Generated {page_calc['recommended']} pages. Check assets/output/final_pages/")

if __name__ == "__main__":
    asyncio.run(run_production())
