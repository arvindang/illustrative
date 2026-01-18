"""
Test complete pipeline: Scripting -> Illustration -> Composition with validation.

This test runs all three agents with comprehensive validation:
1. ScriptingAgent: Generates script + asset manifest (7-pass enrichment)
2. IllustratorAgent: Generates character references + panel images
3. CompositorAgent: Composes pages + exports to final_pages/

Validation hooks at each step:
- Pre-validation: Era anachronisms, dialogue/caption length
- Post-validation: Auto-fixes for common issues
- Continuity validation: Character tracking across panels
- Consistency audit: Cross-panel character appearance checks
- Composition analysis: Bubble placement, negative space

Usage:
    # Default (test mode, 1 page)
    pytest tests/test_full_pipeline.py -v

    # Custom configuration
    pytest tests/test_full_pipeline.py --style "Lush Watercolor" --pages 5 -v

    # Full production mode
    pytest tests/test_full_pipeline.py --full-mode --pages 10 -v

    # Skip validation for faster testing
    pytest tests/test_full_pipeline.py --skip-validation -v

Note: This test makes API calls for image generation and will incur costs.
Use test_scripting_only.py for cheaper iteration on script generation.
"""
import asyncio
import json
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List

import pytest

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents import ScriptingAgent, IllustratorAgent, CompositorAgent
from validators import (
    ScriptValidator,
    ConsistencyAuditor,
    ContinuityValidator,
    ImageCompositionAnalyzer,
    validate_script_continuity,
    create_pre_validator_from_assets,
)


def pytest_addoption(parser):
    """Add pipeline-specific CLI options."""
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


@dataclass
class PipelineMetrics:
    """Collects metrics from pipeline execution."""
    # Generation metrics
    pages_generated: int = 0
    panels_generated: int = 0
    characters_count: int = 0
    references_generated: int = 0
    final_pages_composed: int = 0

    # Validation metrics
    pre_validation_issues: int = 0
    script_auto_fixes: int = 0
    script_manual_reviews: int = 0
    continuity_errors: int = 0
    continuity_warnings: int = 0
    consistency_issues: int = 0
    composition_warnings: int = 0

    # Quality metrics
    avg_panels_per_page: float = 0.0
    dialogue_length_violations: int = 0
    era_anachronisms: int = 0

    def compute_derived(self):
        """Compute derived metrics."""
        if self.pages_generated > 0:
            self.avg_panels_per_page = self.panels_generated / self.pages_generated

    def report(self) -> str:
        """Generate summary report."""
        self.compute_derived()
        lines = [
            "",
            "=" * 60,
            "PIPELINE EXECUTION METRICS",
            "=" * 60,
            "",
            "GENERATION:",
            f"  Pages:                 {self.pages_generated}",
            f"  Panels:                {self.panels_generated}",
            f"  Avg Panels/Page:       {self.avg_panels_per_page:.1f}",
            f"  Characters:            {self.characters_count}",
            f"  References Generated:  {self.references_generated}",
            f"  Final Pages:           {self.final_pages_composed}",
            "",
            "VALIDATION:",
            f"  Pre-validation Issues: {self.pre_validation_issues}",
            f"  Script Auto-fixes:     {self.script_auto_fixes}",
            f"  Manual Review Needed:  {self.script_manual_reviews}",
            f"  Continuity Errors:     {self.continuity_errors}",
            f"  Continuity Warnings:   {self.continuity_warnings}",
            f"  Consistency Issues:    {self.consistency_issues}",
            f"  Composition Warnings:  {self.composition_warnings}",
            "",
            "QUALITY:",
            f"  Dialogue Violations:   {self.dialogue_length_violations}",
            f"  Era Anachronisms:      {self.era_anachronisms}",
            "",
            "=" * 60,
        ]
        return "\n".join(lines)


class TestFullPipeline:
    """Test suite for complete 3-step pipeline with validation."""

    @pytest.mark.asyncio
    async def test_full_pipeline(
        self, input_file, style, tone, test_mode, target_pages, script_path, skip_validation
    ):
        """
        Test complete pipeline from text to final composed pages.

        Steps:
        1. ScriptingAgent: Script + asset manifest
        2. Validation: Pre-validation + post-validation + continuity
        3. IllustratorAgent: Reference sheets + panel images
        4. Validation: Consistency audit
        5. CompositorAgent: Page composition + export
        6. Validation: Final quality checks
        """
        metrics = PipelineMetrics()

        print(f"\n{'='*60}")
        print("FULL PIPELINE TEST WITH VALIDATION")
        print(f"{'='*60}")
        print(f"Input:           {input_file}")
        print(f"Style:           {style}")
        print(f"Tone:            {tone}")
        print(f"Test Mode:       {test_mode}")
        print(f"Pages:           {target_pages}")
        print(f"Skip Validation: {skip_validation}")
        print(f"{'='*60}\n")

        input_stem = Path(input_file).stem
        assets_path = f"assets/output/{input_stem}_assets.json"

        # ===================================================================
        # STEP 1: SCRIPTING
        # ===================================================================
        print("\n--- STEP 1: SCRIPTING + ASSET ANALYSIS ---")
        scripter = ScriptingAgent(input_file)
        script = await scripter.generate_script(
            style=f"{style}, {tone}",
            test_mode=test_mode,
            target_page_override=target_pages if not test_mode else None
        )

        assert script is not None, "Script generation failed"
        metrics.pages_generated = len(script)
        metrics.panels_generated = sum(len(p.get('panels', [])) for p in script)

        # Collect character count
        all_chars = set()
        for page in script:
            for panel in page.get('panels', []):
                all_chars.update(panel.get('characters', []))
        metrics.characters_count = len(all_chars)

        print(f"Script generated: {metrics.pages_generated} page(s), {metrics.panels_generated} panels")

        # ===================================================================
        # STEP 2: VALIDATION (Pre + Post + Continuity)
        # ===================================================================
        if not skip_validation:
            print("\n--- STEP 2: VALIDATION SUITE ---")

            # Load assets for validation
            assets_file = Path(assets_path)
            assets = {}
            if assets_file.exists():
                with open(assets_file, 'r') as f:
                    assets = json.load(f)

            # Determine era
            era = self._detect_era(input_stem)

            # 2a. Pre-validation
            print("\n  [2a] Pre-validation...")
            pre_validator = create_pre_validator_from_assets(era=era, assets=assets)

            for page in script:
                for panel in page.get('panels', []):
                    result = pre_validator.validate_panel_inputs(
                        visual_description=panel.get('visual_description', ''),
                        dialogue=panel.get('dialogue', ''),
                        caption=panel.get('caption', ''),
                        characters=panel.get('characters', [])
                    )
                    if not result.valid:
                        metrics.pre_validation_issues += 1
                        for issue in result.issues:
                            if "anachronism" in issue.lower():
                                metrics.era_anachronisms += 1
                            if "dialogue" in issue.lower() or "caption" in issue.lower():
                                metrics.dialogue_length_violations += 1

            print(f"       Pre-validation issues: {metrics.pre_validation_issues}")

            # 2b. Post-validation with auto-fixes
            print("\n  [2b] Post-validation + auto-fix...")
            post_validator = ScriptValidator(era=era, assets=assets)
            fixed_script, report = post_validator.validate_and_fix(script)

            metrics.script_auto_fixes = report.auto_fixed
            metrics.script_manual_reviews = report.manual_review

            print(f"       Auto-fixes applied: {report.auto_fixed}")
            print(f"       Manual review needed: {report.manual_review}")

            if report.script_modified:
                with open(script_path, 'w') as f:
                    json.dump(fixed_script, f, indent=2)
                script = fixed_script
                print("       Fixed script saved.")

            # 2c. Continuity validation
            print("\n  [2c] Continuity validation...")
            char_dir = Path("assets/output/characters")
            continuity_result = validate_script_continuity(script_path, str(char_dir))

            metrics.continuity_errors = len(continuity_result.get('errors', []))
            metrics.continuity_warnings = len(continuity_result.get('warnings', []))

            print(f"       Continuity errors: {metrics.continuity_errors}")
            print(f"       Continuity warnings: {metrics.continuity_warnings}")

        # ===================================================================
        # STEP 3: ILLUSTRATION
        # ===================================================================
        print("\n--- STEP 3: ILLUSTRATION ---")
        illustrator = IllustratorAgent(script_path, f"{style} style, {tone} tone")

        # Generate reference sheets
        print("\n  [3a] Generating reference sheets...")
        await illustrator.generate_all_references(style=style)

        # Count generated references
        char_dir = Path("assets/output/characters")
        if char_dir.exists():
            for folder in char_dir.iterdir():
                if folder.is_dir() and list(folder.glob("*.png")):
                    metrics.references_generated += 1

        print(f"       References generated: {metrics.references_generated}")

        # Generate panel images
        print("\n  [3b] Generating panel images...")
        await illustrator.run_production()

        # Verify panel images
        pages_dir = Path("assets/output/pages")
        panel_images_count = 0
        if pages_dir.exists():
            panel_images_count = len(list(pages_dir.rglob("*.png")))
        print(f"       Panel images: {panel_images_count}")

        # ===================================================================
        # STEP 4: CONSISTENCY AUDIT (if not skipped)
        # ===================================================================
        if not skip_validation and panel_images_count > 0:
            print("\n--- STEP 4: CONSISTENCY AUDIT ---")
            metrics.consistency_issues = await self._run_consistency_audit(script)
            print(f"       Consistency issues: {metrics.consistency_issues}")

        # ===================================================================
        # STEP 5: COMPOSITION
        # ===================================================================
        print("\n--- STEP 5: COMPOSITION + EXPORT ---")
        compositor = CompositorAgent(script_path)
        compositor.run()

        # Verify final pages
        final_pages_dir = Path("assets/output/final_pages")
        if final_pages_dir.exists():
            final_pages = list(final_pages_dir.glob("*.png"))
            metrics.final_pages_composed = len(final_pages)
        print(f"       Final pages composed: {metrics.final_pages_composed}")

        assert metrics.final_pages_composed > 0, "No final pages were generated"

        # ===================================================================
        # STEP 6: COMPOSITION QUALITY CHECK (if not skipped)
        # ===================================================================
        if not skip_validation and metrics.final_pages_composed > 0:
            print("\n--- STEP 6: COMPOSITION QUALITY CHECK ---")
            # Basic check: final pages should be larger than panel images
            # (composition adds borders, bubbles, etc.)
            final_page = final_pages_dir / "page_1.png"
            if final_page.exists():
                from PIL import Image
                with Image.open(final_page) as img:
                    width, height = img.size
                    print(f"       Final page size: {width}x{height}")
                    # Reasonable size check
                    if width < 800 or height < 1000:
                        metrics.composition_warnings += 1
                        print("       Warning: Final page seems small")

        # ===================================================================
        # FINAL REPORT
        # ===================================================================
        print(metrics.report())

        # Final assertions
        assert metrics.pages_generated > 0, "No pages generated"
        assert metrics.final_pages_composed > 0, "No final pages composed"

        if not skip_validation:
            assert metrics.continuity_errors == 0, \
                f"Continuity errors found: {metrics.continuity_errors}"
            assert metrics.script_manual_reviews <= 2, \
                f"Too many items needing manual review: {metrics.script_manual_reviews}"

        print(f"\nPipeline complete! Check: assets/output/final_pages/")

    async def _run_consistency_audit(self, script: List[Dict]) -> int:
        """Run consistency audit on generated panels, return issue count."""
        from PIL import Image

        auditor = ConsistencyAuditor()
        pages_dir = Path("assets/output/pages")
        total_issues = 0

        if not pages_dir.exists():
            return 0

        # Audit first 2 pages
        for page in script[:2]:
            page_num = page.get('page_number', 0)
            page_dir = pages_dir / f"page_{page_num}"

            if not page_dir.exists():
                continue

            panel_images = sorted(page_dir.glob("panel_*.png"))
            if len(panel_images) < 2:
                continue

            images = []
            for img_path in panel_images[:4]:
                try:
                    images.append(Image.open(img_path))
                except Exception:
                    pass

            if len(images) < 2:
                continue

            chars_on_page = set()
            for panel in page.get('panels', []):
                chars_on_page.update(panel.get('characters', []))

            result = await auditor.audit_page_consistency(
                page_images=images,
                characters=list(chars_on_page)
            )

            if not result.consistent:
                total_issues += len(result.character_issues)
                print(f"       Page {page_num}: Issues detected")
            else:
                print(f"       Page {page_num}: Consistent")

            for img in images:
                img.close()

        return total_issues

    def _detect_era(self, input_stem: str) -> str:
        """Detect era from input file name or global context."""
        # Check for global context file
        context_file = Path(f"assets/output/{input_stem}_global_context.json")
        if context_file.exists():
            with open(context_file, 'r') as f:
                context = json.load(f)
                return context.get('era_setting', '')

        # Fallback: detect from filename
        if "league" in input_stem.lower() or "20" in input_stem:
            return "1860s Victorian"
        if "alice" in input_stem.lower():
            return "1860s Victorian"
        return ""

    @pytest.mark.asyncio
    async def test_resume_from_script(self, script_path, style, tone, skip_validation):
        """
        Test resuming pipeline from existing script JSON.

        Useful when you already have a script and just want to re-run
        illustration and composition steps.
        """
        script_file = Path(script_path)
        if not script_file.exists():
            pytest.skip(f"Script file not found: {script_path}. Run test_generate_script first.")

        print(f"\n{'='*60}")
        print("RESUME FROM SCRIPT TEST")
        print(f"{'='*60}")
        print(f"Script:  {script_path}")
        print(f"Style:   {style}")
        print(f"{'='*60}\n")

        # Load script for validation
        with open(script_file, 'r') as f:
            script = json.load(f)

        # Optional: Run post-validation on existing script
        if not skip_validation:
            print("\n--- VALIDATION: Post-validation ---")
            input_stem = Path(script_path).stem.replace("_test_page", "").replace("_full_script", "")
            era = self._detect_era(input_stem)
            post_validator = ScriptValidator(era=era)
            fixed_script, report = post_validator.validate_and_fix(script)
            print(f"       Auto-fixes: {report.auto_fixed}")
            print(f"       Manual review: {report.manual_review}")

            if report.script_modified:
                with open(script_file, 'w') as f:
                    json.dump(fixed_script, f, indent=2)

        # --- STEP 2: ILLUSTRATION ---
        print("\n--- STEP 2: ILLUSTRATION (Resume) ---")
        illustrator = IllustratorAgent(script_path, f"{style} style, {tone} tone")
        await illustrator.generate_all_references(style=style)
        await illustrator.run_production()

        # --- STEP 3: COMPOSITION ---
        print("\n--- STEP 3: COMPOSITION (Resume) ---")
        compositor = CompositorAgent(script_path)
        compositor.run()

        final_pages_dir = Path("assets/output/final_pages")
        final_pages = list(final_pages_dir.glob("*.png")) if final_pages_dir.exists() else []
        print(f"\nResume complete! {len(final_pages)} pages in assets/output/final_pages/")

        assert len(final_pages) > 0, "No final pages generated"


class TestPipelineValidationOnly:
    """Tests that run validation without generation (uses existing outputs)."""

    @pytest.mark.asyncio
    async def test_validate_existing_script(self, script_path):
        """Validate an existing script without regenerating."""
        script_file = Path(script_path)
        if not script_file.exists():
            pytest.skip(f"Script not found: {script_path}")

        print(f"\n{'='*60}")
        print("VALIDATE EXISTING SCRIPT")
        print(f"{'='*60}\n")

        with open(script_file, 'r') as f:
            script = json.load(f)

        # Post-validation
        validator = ScriptValidator(era="1860s")
        fixed_script, report = validator.validate_and_fix(script)

        print(f"Pages: {len(script)}")
        print(f"Panels: {sum(len(p['panels']) for p in script)}")
        print(f"Auto-fixes: {report.auto_fixed}")
        print(f"Manual review: {report.manual_review}")
        print(f"\n{validator.get_report_summary()}")

        assert report.manual_review <= 2, f"Script needs manual review"

    @pytest.mark.asyncio
    async def test_validate_existing_references(self):
        """Validate existing reference sheets for consistency."""
        char_dir = Path("assets/output/characters")
        if not char_dir.exists():
            pytest.skip("No character directory found")

        print(f"\n{'='*60}")
        print("VALIDATE EXISTING REFERENCES")
        print(f"{'='*60}\n")

        issues = []
        for folder in char_dir.iterdir():
            if not folder.is_dir():
                continue

            # Check metadata exists
            metadata_path = folder / "metadata.json"
            if not metadata_path.exists():
                issues.append(f"{folder.name}: Missing metadata.json")
                continue

            # Check at least one reference image
            refs = list(folder.glob("*.png"))
            if not refs:
                issues.append(f"{folder.name}: No reference images")
                continue

            print(f"  [OK] {folder.name}: {len(refs)} reference(s)")

        if issues:
            print(f"\nIssues found: {len(issues)}")
            for issue in issues:
                print(f"  - {issue}")

        # Allow some issues but not all
        char_count = len([f for f in char_dir.iterdir() if f.is_dir()])
        assert len(issues) < char_count, f"Too many reference issues: {issues}"


# Allow running as standalone script
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run full pipeline test")
    parser.add_argument("--input-file", default="assets/input/20-thousand-leagues-under-the-sea.txt")
    parser.add_argument("--style", default="Manga/Anime")
    parser.add_argument("--tone", default="Philosophical")
    parser.add_argument("--pages", type=int, default=1)
    parser.add_argument("--full-mode", action="store_true")
    parser.add_argument("--skip-validation", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Resume from existing script")
    args = parser.parse_args()

    test_mode = not args.full_mode
    input_stem = Path(args.input_file).stem
    suffix = "_test_page" if test_mode else "_full_script"
    script_path = f"assets/output/{input_stem}{suffix}.json"

    async def run():
        test = TestFullPipeline()
        if args.resume:
            await test.test_resume_from_script(script_path, args.style, args.tone, args.skip_validation)
        else:
            await test.test_full_pipeline(
                args.input_file, args.style, args.tone,
                test_mode, args.pages, script_path, args.skip_validation
            )

    asyncio.run(run())
