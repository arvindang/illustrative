"""
Quick smoke test for the complete pipeline with validation hooks.

This is a minimal test that runs all 3 steps with default settings
to verify the pipeline is working end-to-end, with built-in quality checks.

Usage:
    # Run via pytest
    pytest tests/test_smoke.py -v

    # Run as standalone script
    python tests/test_smoke.py

    # Skip validation (faster, for quick checks)
    pytest tests/test_smoke.py --skip-validation -v
"""
import asyncio
import json
import sys
from pathlib import Path

import pytest

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents import ScriptingAgent, IllustratorAgent, CompositorAgent
from validators import (
    ScriptValidator,
    ConsistencyAuditor,
    validate_script_continuity,
    create_pre_validator_from_assets,
)


def pytest_addoption(parser):
    """Add smoke test specific options."""
    try:
        parser.addoption(
            "--skip-validation",
            action="store_true",
            default=False,
            help="Skip validation steps for faster testing"
        )
    except ValueError:
        pass


@pytest.fixture
def skip_validation(request):
    """Whether to skip validation steps."""
    try:
        return request.config.getoption("--skip-validation")
    except ValueError:
        return False


@pytest.mark.asyncio
async def test_smoke(skip_validation=False):
    """
    Simplified 3-step smoke test with validation:
    1. ScriptingAgent: Script + asset manifest
    2. IllustratorAgent: Reference sheets + panels
    3. CompositorAgent: Compose + export

    Validation hooks:
    - Pre-validation: Era anachronisms, dialogue length
    - Post-validation: Script auto-fixes
    - Continuity validation: Character tracking
    - Consistency audit: Cross-panel character consistency
    """
    print("\nStarting Smoke Test (3-Step Pipeline with Validation)...")

    input_file = "assets/input/20-thousand-leagues-under-the-sea.txt"
    style = "Manga/Anime"
    tone = "Philosophical"
    input_stem = Path(input_file).stem

    script_path = f"assets/output/{input_stem}_test_page.json"
    assets_path = f"assets/output/{input_stem}_assets.json"

    validation_results = {
        "pre_validation_issues": 0,
        "script_auto_fixes": 0,
        "continuity_errors": 0,
        "continuity_warnings": 0,
        "consistency_passed": True,
    }

    # --- STEP 1: SCRIPTING (includes asset manifest) ---
    print("\n--- STEP 1: SCRIPTING + ASSET ANALYSIS ---")
    scripter = ScriptingAgent(input_file)
    script = await scripter.generate_script(style=style, test_mode=True)

    assert script is not None, "Script generation failed"
    print(f"Script generated: {len(script)} page(s)")

    # --- VALIDATION HOOK: Pre-validation ---
    if not skip_validation:
        print("\n--- VALIDATION: Pre-validation ---")
        assets_file = Path(assets_path)
        assets = {}
        if assets_file.exists():
            with open(assets_file, 'r') as f:
                assets = json.load(f)

        pre_validator = create_pre_validator_from_assets(
            era="1860s",  # 20k Leagues setting
            assets=assets
        )

        for page in script:
            for panel in page.get('panels', []):
                result = pre_validator.validate_panel_inputs(
                    visual_description=panel.get('visual_description', ''),
                    dialogue=panel.get('dialogue', ''),
                    caption=panel.get('caption', ''),
                    characters=panel.get('characters', [])
                )
                if not result.valid:
                    validation_results["pre_validation_issues"] += 1

        print(f"Pre-validation issues: {validation_results['pre_validation_issues']}")

    # --- VALIDATION HOOK: Post-validation with auto-fixes ---
    if not skip_validation:
        print("\n--- VALIDATION: Post-validation + Auto-fix ---")
        post_validator = ScriptValidator(era="1860s")
        fixed_script, report = post_validator.validate_and_fix(script)

        validation_results["script_auto_fixes"] = report.auto_fixed
        print(f"Auto-fixes applied: {report.auto_fixed}")

        if report.script_modified:
            # Save the fixed script
            with open(script_path, 'w') as f:
                json.dump(fixed_script, f, indent=2)
            script = fixed_script
            print("Fixed script saved.")

    # --- VALIDATION HOOK: Continuity validation ---
    if not skip_validation:
        print("\n--- VALIDATION: Continuity ---")
        char_dir = Path("assets/output/characters")
        continuity_result = validate_script_continuity(script_path, str(char_dir))

        validation_results["continuity_errors"] = len(continuity_result.get('errors', []))
        validation_results["continuity_warnings"] = len(continuity_result.get('warnings', []))

        print(f"Continuity errors: {validation_results['continuity_errors']}")
        print(f"Continuity warnings: {validation_results['continuity_warnings']}")

    # --- STEP 2: ILLUSTRATION (reference sheets + panels) ---
    print("\n--- STEP 2: ILLUSTRATION ---")
    illustrator = IllustratorAgent(script_path, f"{style} style, {tone} tone")

    # Generate reference sheets for characters/objects
    await illustrator.generate_all_references(style=style)

    # Generate panel images
    await illustrator.run_production()

    # Verify panels generated
    pages_dir = Path("assets/output/pages")
    panel_count = 0
    if pages_dir.exists():
        panel_count = len(list(pages_dir.rglob("*.png")))
    print(f"Panel images generated: {panel_count}")

    # --- VALIDATION HOOK: Consistency audit ---
    if not skip_validation and panel_count > 0:
        print("\n--- VALIDATION: Consistency Audit ---")
        await _run_consistency_audit(script, validation_results)

    # --- STEP 3: COMPOSITION + EXPORT ---
    print("\n--- STEP 3: COMPOSITION + EXPORT ---")
    compositor = CompositorAgent(script_path)
    compositor.run()

    # Verify final output
    final_pages_dir = Path("assets/output/final_pages")
    final_count = 0
    if final_pages_dir.exists():
        final_count = len(list(final_pages_dir.glob("*.png")))
    print(f"Final pages composed: {final_count}")

    # --- FINAL REPORT ---
    print("\n" + "=" * 60)
    print("SMOKE TEST RESULTS")
    print("=" * 60)
    print(f"Pages:       {len(script)}")
    print(f"Panels:      {panel_count}")
    print(f"Final Pages: {final_count}")
    if not skip_validation:
        print(f"\nValidation:")
        print(f"  Pre-validation issues:  {validation_results['pre_validation_issues']}")
        print(f"  Script auto-fixes:      {validation_results['script_auto_fixes']}")
        print(f"  Continuity errors:      {validation_results['continuity_errors']}")
        print(f"  Continuity warnings:    {validation_results['continuity_warnings']}")
        print(f"  Consistency passed:     {validation_results['consistency_passed']}")
    print("=" * 60)

    # Assertions
    assert final_count > 0, "No final pages generated"
    if not skip_validation:
        assert validation_results["continuity_errors"] == 0, \
            f"Continuity errors found: {validation_results['continuity_errors']}"

    print("\nSmoke Test Complete! Check assets/output/final_pages/")


async def _run_consistency_audit(script, validation_results):
    """Run consistency audit on generated panels."""
    from PIL import Image

    auditor = ConsistencyAuditor()
    pages_dir = Path("assets/output/pages")

    if not pages_dir.exists():
        print("  Skipping - no pages directory")
        return

    # Check first page only for smoke test
    for page in script[:1]:
        page_num = page.get('page_number', 0)
        page_dir = pages_dir / f"page_{page_num}"

        if not page_dir.exists():
            continue

        panel_images = sorted(page_dir.glob("panel_*.png"))
        if len(panel_images) < 2:
            continue

        # Load images
        images = []
        for img_path in panel_images[:3]:  # Max 3 for speed
            try:
                images.append(Image.open(img_path))
            except Exception:
                pass

        if len(images) < 2:
            continue

        # Get characters
        chars_on_page = set()
        for panel in page.get('panels', []):
            chars_on_page.update(panel.get('characters', []))

        result = await auditor.audit_page_consistency(
            page_images=images,
            characters=list(chars_on_page)
        )

        if result.consistent:
            print(f"  Page {page_num}: Consistent")
        else:
            print(f"  Page {page_num}: Issues detected")
            validation_results["consistency_passed"] = False

        # Cleanup
        for img in images:
            img.close()


@pytest.mark.asyncio
async def test_smoke_script_only():
    """
    Smoke test for script generation only (no images).
    Faster test that validates the scripting pipeline.
    """
    print("\nStarting Script-Only Smoke Test...")

    input_file = "assets/input/20-thousand-leagues-under-the-sea.txt"
    style = "Manga/Anime"
    input_stem = Path(input_file).stem
    script_path = f"assets/output/{input_stem}_test_page.json"
    assets_path = f"assets/output/{input_stem}_assets.json"

    # Generate script
    scripter = ScriptingAgent(input_file)
    script = await scripter.generate_script(style=style, test_mode=True)

    assert script is not None, "Script generation failed"
    assert len(script) > 0, "Script is empty"

    # Basic structure validation
    for page in script:
        assert "page_number" in page, "Page missing page_number"
        assert "panels" in page, "Page missing panels"
        for panel in page["panels"]:
            assert "panel_id" in panel, "Panel missing panel_id"
            assert "visual_description" in panel, "Panel missing visual_description"

    # Post-validation
    post_validator = ScriptValidator(era="1860s")
    fixed_script, report = post_validator.validate_and_fix(script)

    print(f"\nScript-Only Results:")
    print(f"  Pages: {len(script)}")
    print(f"  Panels: {sum(len(p['panels']) for p in script)}")
    print(f"  Auto-fixes: {report.auto_fixed}")
    print(f"  Manual review: {report.manual_review}")

    assert report.manual_review == 0, f"Script needs manual review: {report.manual_review} items"

    print("\nScript-Only Smoke Test Complete!")


# Allow running as standalone script
if __name__ == "__main__":
    asyncio.run(test_smoke())
