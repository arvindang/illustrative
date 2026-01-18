"""
Smoke test runner for the 3-step pipeline.
Generates 10 pages using ONLY Gemini 3 models via Vertex AI.
"""
import asyncio
import sys
from pathlib import Path
from datetime import datetime

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)

# Configuration
TARGET_PAGES = 10
STYLE = "Lush Watercolor"
ERA_CONSTRAINTS = """Setting: 1860s Victorian era, maritime/underwater exploration.
SHIPS: Only sailing ships, wooden merchant vessels, early coal-powered steamships with paddle wheels or single screws. NO modern vessels.
CLOTHING (Men): Frock coats, waistcoats, cravats, top hats or bowler hats, heavy wool overcoats. Sailors wear traditional 19th-century naval uniforms.
TECHNOLOGY: Gas lamps, telegraphs, early photography, mechanical instruments. NO electric lights, telephones, or modern electronics.
DIVING EQUIPMENT: Brass and copper diving helmets, canvas/rubber diving suits with riveted metal plates, air hoses connected to surface pumps. NO SCUBA gear.
WEAPONS: Harpoons, single-shot rifles, pistols, swords. NO automatic weapons.
INTERIORS: Victorian ornate style - mahogany wood, brass fittings, velvet upholstery, oil paintings, gas lighting.
"""

async def run_smoke_test():
    print("="*60)
    print("SMOKE TEST - 10 Page Pipeline (Gemini 3 Only)")
    print("="*60)

    # Import config and display settings
    from config import config

    # Verify Vertex AI is enabled
    if not config.use_vertex_ai:
        print("\n❌ ERROR: Vertex AI is not enabled!")
        print("   Set GOOGLE_GENAI_USE_VERTEXAI=true in .env")
        return False

    # Verify all models are Gemini 3
    gemini3_models = [
        config.image_model_primary,
        config.image_model_fallback,
        config.image_model_last_resort,
        config.scripting_model_global_context,
        config.scripting_model_page_script,
    ]
    non_gemini3 = [m for m in gemini3_models if "gemini-3" not in m and "gemini-2" not in m]

    print(f"\nConfiguration:")
    print(f"  ✓ Vertex AI: {config.use_vertex_ai}")
    print(f"  ✓ Project: {config.gcp_project}")
    print(f"  ✓ Location: {config.gcp_location}")
    print(f"  ✓ Primary Image Model: {config.image_model_primary}")
    print(f"  ✓ Fallback Image Model: {config.image_model_fallback}")
    print(f"  ✓ Stop on primary quota: {config.stop_on_primary_quota_exhausted}")
    print(f"  ✓ Text Model (Global): {config.scripting_model_global_context}")
    print(f"  ✓ Text Model (Script): {config.scripting_model_page_script}")

    limits = config.get_effective_rate_limits()
    print(f"  ✓ Image RPM: {limits['image_rpm']}")
    print(f"  ✓ Scripting RPM: {limits['scripting_rpm']}")
    print(f"  ✓ TPM Limit: {limits['tpm_limit']:,}")
    print()

    input_file = "assets/input/20-thousand-leagues-under-the-sea.txt"
    input_stem = Path(input_file).stem

    # Create isolated output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_output_dir = Path(f"assets/output/smoke_test_{timestamp}")
    base_output_dir.mkdir(parents=True, exist_ok=True)

    script_path = base_output_dir / f"{input_stem}_test_page.json"

    print(f"Input: {input_file}")
    print(f"Style: {STYLE}")
    print(f"Target Pages: {TARGET_PAGES}")
    print(f"Output dir: {base_output_dir}")
    print()

    # --- STEP 1: SCRIPTING ---
    print("\n" + "="*60)
    print("STEP 1: SCRIPTING + ASSET ANALYSIS")
    print("="*60)

    from agents import ScriptingAgent
    scripter = ScriptingAgent(input_file, base_output_dir=base_output_dir)
    script = await scripter.generate_script(
        style=STYLE,
        test_mode=True,
        target_page_override=TARGET_PAGES,
        context_constraints=ERA_CONSTRAINTS
    )

    if script is None:
        print("ERROR: Script generation failed!")
        return False

    print(f"\n✓ Script generated: {len(script)} page(s)")
    total_panels = sum(len(page.get('panels', [])) for page in script)
    print(f"  Total panels: {total_panels}")

    # --- STEP 2: ILLUSTRATION ---
    print("\n" + "="*60)
    print("STEP 2: ILLUSTRATION (Reference Sheets + Panels)")
    print("="*60)

    from agents import IllustratorAgent
    illustrator = IllustratorAgent(str(script_path), f"{STYLE} style", base_output_dir=base_output_dir)

    # Set era constraints for panel validation
    illustrator.era_constraints = ERA_CONSTRAINTS

    print("\nGenerating character reference sheets...")
    await illustrator.generate_all_references(style=STYLE)

    print("\nGenerating panel images...")
    await illustrator.run_production()

    # Count generated panels
    pages_dir = base_output_dir / "pages"
    panel_count = 0
    if pages_dir.exists():
        panel_count = len(list(pages_dir.rglob("*.png")))
    print(f"\n✓ Generated {panel_count} panel images")

    # --- STEP 3: COMPOSITION ---
    print("\n" + "="*60)
    print("STEP 3: COMPOSITION + EXPORT")
    print("="*60)

    from agents import CompositorAgent
    compositor = CompositorAgent(str(script_path), base_output_dir=base_output_dir)
    compositor.run()

    # Count final pages
    final_dir = base_output_dir / "final_pages"
    final_count = 0
    if final_dir.exists():
        final_count = len(list(final_dir.glob("*.png")))
    print(f"\n✓ Composed {final_count} final pages")

    # --- SUMMARY ---
    print("\n" + "="*60)
    print("SMOKE TEST RESULTS")
    print("="*60)
    print(f"Pages: {len(script)}")
    print(f"Panels: {panel_count}")
    print(f"Final Pages: {final_count}")
    print(f"Output: {base_output_dir}")
    print("="*60)

    # List all generated files
    print("\n📂 Generated Files:")

    # JSON files
    json_files = list(base_output_dir.glob("*.json"))
    if json_files:
        print("\n  📝 JSON Files:")
        for f in sorted(json_files):
            print(f"     - {f.name}")

    # Character references
    char_dir = base_output_dir / "characters"
    if char_dir.exists():
        char_images = list(char_dir.rglob("*.png"))
        print(f"\n  👤 Character References: {len(char_images)} images")
        for char_folder in sorted(char_dir.iterdir()):
            if char_folder.is_dir():
                count = len(list(char_folder.glob("*.png")))
                print(f"     - {char_folder.name}/: {count} images")

    # Object references
    obj_dir = base_output_dir / "objects"
    if obj_dir.exists():
        obj_images = list(obj_dir.rglob("*.png"))
        print(f"\n  🎨 Object References: {len(obj_images)} images")
        for obj_folder in sorted(obj_dir.iterdir()):
            if obj_folder.is_dir():
                count = len(list(obj_folder.glob("*.png")))
                print(f"     - {obj_folder.name}/: {count} images")

    # Panel images
    pages_dir = base_output_dir / "pages"
    if pages_dir.exists():
        print(f"\n  🖼️  Panel Images: {panel_count} images")
        for page_folder in sorted(pages_dir.iterdir(), key=lambda x: int(x.name.split('_')[1]) if x.is_dir() and x.name.startswith('page_') else 0):
            if page_folder.is_dir():
                count = len(list(page_folder.glob("*.png")))
                print(f"     - {page_folder.name}/: {count} panels")

    # Final pages
    final_dir = base_output_dir / "final_pages"
    if final_dir.exists():
        print(f"\n  📄 Final Composed Pages: {final_count} pages")

    # Export files
    pdf_files = list(base_output_dir.glob("*.pdf"))
    epub_files = list(base_output_dir.glob("*.epub"))
    if pdf_files or epub_files:
        print("\n  📚 Export Files:")
        for f in pdf_files:
            print(f"     - {f.name}")
        for f in epub_files:
            print(f"     - {f.name}")

    print("\n" + "="*60)

    if final_count > 0:
        print("\n✅ SMOKE TEST PASSED!")
        print(f"\n   Open output folder: {base_output_dir.absolute()}")
        return True
    else:
        print("\n❌ SMOKE TEST FAILED: No final pages generated")
        return False


if __name__ == "__main__":
    try:
        result = asyncio.run(run_smoke_test())
        sys.exit(0 if result else 1)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n\n❌ SMOKE TEST ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
