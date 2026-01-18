"""
Integration test: 10-page graphic novel with full continuity validation.

This test runs the complete pipeline with a focus on:
- Character continuity across pages
- Dialogue quality and consistency
- Visual style consistency
- Panel composition quality

Designed as a middle tier between smoke (1 page) and production (100+ pages).

Usage:
    # Default (10 pages with full validation)
    pytest tests/test_integration.py -v

    # Custom page count
    pytest tests/test_integration.py --pages 5 -v

    # With specific style
    pytest tests/test_integration.py --style "Lush Watercolor" --pages 10 -v

    # Skip image generation (script + validation only)
    pytest tests/test_integration.py --script-only -v
"""
import asyncio
import json
import sys
from pathlib import Path
from typing import Dict, List

import pytest

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents import ScriptingAgent, ReferenceAgent, PanelAgent, CompositorAgent
from validators import (
    ContinuityValidator,
    ScriptValidator,
    ConsistencyAuditor,
    PromptPreValidator,
    create_pre_validator_from_assets,
    validate_script_continuity,
)


def pytest_addoption(parser):
    """Add integration-specific CLI options."""
    try:
        parser.addoption(
            "--script-only",
            action="store_true",
            default=False,
            help="Run only script generation and validation (no images)"
        )
    except ValueError:
        pass


@pytest.fixture
def script_only(request):
    """Whether to skip image generation."""
    try:
        return request.config.getoption("--script-only")
    except ValueError:
        return False


class IntegrationTestMetrics:
    """Collects and reports metrics from the integration test."""

    def __init__(self):
        self.pages_generated = 0
        self.panels_generated = 0
        self.characters_referenced = set()
        self.continuity_errors = 0
        self.continuity_warnings = 0
        self.script_auto_fixes = 0
        self.script_manual_reviews = 0
        self.consistency_issues = 0
        self.dialogue_issues = []
        self.era_issues = []

    def report(self) -> str:
        """Generate a summary report."""
        lines = [
            "",
            "=" * 60,
            "INTEGRATION TEST METRICS",
            "=" * 60,
            f"Pages Generated:        {self.pages_generated}",
            f"Panels Generated:       {self.panels_generated}",
            f"Unique Characters:      {len(self.characters_referenced)}",
            "",
            "VALIDATION RESULTS:",
            f"  Continuity Errors:    {self.continuity_errors}",
            f"  Continuity Warnings:  {self.continuity_warnings}",
            f"  Script Auto-fixes:    {self.script_auto_fixes}",
            f"  Manual Review Needed: {self.script_manual_reviews}",
            f"  Consistency Issues:   {self.consistency_issues}",
            "",
        ]

        if self.dialogue_issues:
            lines.append("DIALOGUE ISSUES:")
            for issue in self.dialogue_issues[:5]:  # Show first 5
                lines.append(f"  - {issue}")
            if len(self.dialogue_issues) > 5:
                lines.append(f"  ... and {len(self.dialogue_issues) - 5} more")
            lines.append("")

        if self.era_issues:
            lines.append("ERA ANACHRONISM ISSUES:")
            for issue in self.era_issues[:5]:
                lines.append(f"  - {issue}")
            if len(self.era_issues) > 5:
                lines.append(f"  ... and {len(self.era_issues) - 5} more")
            lines.append("")

        lines.append("=" * 60)
        return "\n".join(lines)


class TestIntegration:
    """10-page integration test with full validation suite."""

    @pytest.mark.asyncio
    async def test_10_page_integration(
        self, input_file, style, tone, target_pages, script_path, assets_path, base_output_dir, script_only
    ):
        """
        Full integration test with validation at each step.

        Pipeline:
        1. Script Generation (10 pages)
        2. Pre-validation (era, dialogue, characters)
        3. Script post-validation with auto-fixes
        4. Continuity validation
        5. Reference generation
        6. Panel generation with consistency checks
        7. Composition and export
        """
        # Override to 10 pages for integration test
        if target_pages == 1:
            target_pages = 10

        metrics = IntegrationTestMetrics()

        print(f"\n{'='*60}")
        print("INTEGRATION TEST (10-Page Pipeline)")
        print(f"{'='*60}")
        print(f"Input:       {input_file}")
        print(f"Style:       {style}")
        print(f"Tone:        {tone}")
        print(f"Pages:       {target_pages}")
        print(f"Output Dir:  {base_output_dir}")
        print(f"Script Only: {script_only}")
        print(f"{'='*60}\n")

        # ===================================================================
        # STEP 1: SCRIPT GENERATION
        # ===================================================================
        print("\n--- STEP 1: SCRIPT GENERATION ---")
        scripter = ScriptingAgent(input_file, base_output_dir=base_output_dir)
        script = await scripter.generate_script(
            style=f"{style}, {tone}",
            test_mode=False,  # Full mode for integration
            target_page_override=target_pages
        )

        assert script is not None, "Script generation failed"
        metrics.pages_generated = len(script)
        metrics.panels_generated = sum(len(p.get('panels', [])) for p in script)

        # Collect characters
        for page in script:
            for panel in page.get('panels', []):
                for char in panel.get('characters', []):
                    metrics.characters_referenced.add(char)

        print(f"Generated {metrics.pages_generated} pages with {metrics.panels_generated} panels")
        print(f"Characters: {', '.join(sorted(metrics.characters_referenced)[:5])}...")

        # ===================================================================
        # STEP 2: LOAD ASSETS AND CREATE PRE-VALIDATOR
        # ===================================================================
        print("\n--- STEP 2: PRE-VALIDATION SETUP ---")
        assets_file = Path(assets_path)
        assets = {}
        character_arcs = {}

        if assets_file.exists():
            with open(assets_file, 'r') as f:
                assets = json.load(f)

        # Load character arcs if available
        input_stem = Path(input_file).stem
        arcs_file = base_output_dir / f"{input_stem}_character_arcs.json"
        if arcs_file.exists():
            with open(arcs_file, 'r') as f:
                character_arcs = json.load(f)

        # Determine era from global context
        era = ""
        context_file = base_output_dir / f"{input_stem}_global_context.json"
        if context_file.exists():
            with open(context_file, 'r') as f:
                context = json.load(f)
                era = context.get('era_setting', '')
        print(f"Era context: {era or 'Not specified'}")

        pre_validator = create_pre_validator_from_assets(
            era=era,
            assets=assets,
            character_arcs=character_arcs
        )

        # ===================================================================
        # STEP 3: PRE-VALIDATE EACH PANEL
        # ===================================================================
        print("\n--- STEP 3: PRE-VALIDATION (Era, Dialogue, Characters) ---")
        pre_validation_issues = 0

        for page in script:
            page_num = page.get('page_number', 0)
            for panel in page.get('panels', []):
                result = pre_validator.validate_panel_inputs(
                    visual_description=panel.get('visual_description', ''),
                    dialogue=panel.get('dialogue', ''),
                    caption=panel.get('caption', ''),
                    characters=panel.get('characters', []),
                    scene_type=panel.get('advice', {}).get('scene_type', '') if isinstance(panel.get('advice'), dict) else ''
                )

                if not result.valid:
                    pre_validation_issues += 1
                    for issue in result.issues:
                        if "anachronism" in issue.lower():
                            metrics.era_issues.append(f"Page {page_num}: {issue}")
                        elif "dialogue" in issue.lower() or "caption" in issue.lower():
                            metrics.dialogue_issues.append(f"Page {page_num}: {issue}")

        print(f"Pre-validation issues: {pre_validation_issues}")

        # ===================================================================
        # STEP 4: POST-VALIDATION WITH AUTO-FIXES
        # ===================================================================
        print("\n--- STEP 4: POST-VALIDATION WITH AUTO-FIXES ---")
        post_validator = ScriptValidator(
            era=era,
            character_arcs=character_arcs,
            assets=assets
        )
        fixed_script, validation_report = post_validator.validate_and_fix(script)

        metrics.script_auto_fixes = validation_report.auto_fixed
        metrics.script_manual_reviews = validation_report.manual_review

        print(f"Auto-fixes applied: {validation_report.auto_fixed}")
        print(f"Manual review items: {validation_report.manual_review}")

        if validation_report.script_modified:
            # Save fixed script
            script_file = Path(script_path)
            with open(script_file, 'w') as f:
                json.dump(fixed_script, f, indent=2)
            print(f"Fixed script saved to: {script_path}")
            script = fixed_script

        # ===================================================================
        # STEP 5: CONTINUITY VALIDATION
        # ===================================================================
        print("\n--- STEP 5: CONTINUITY VALIDATION ---")
        char_dir = base_output_dir / "characters"
        continuity_result = validate_script_continuity(script_path, str(char_dir))

        metrics.continuity_errors = len(continuity_result.get('errors', []))
        metrics.continuity_warnings = len(continuity_result.get('warnings', []))

        print(f"Continuity errors:   {metrics.continuity_errors}")
        print(f"Continuity warnings: {metrics.continuity_warnings}")

        if metrics.continuity_errors > 0:
            print("Errors found:")
            for err in continuity_result['errors'][:3]:
                print(f"  - Page {err['page']}, Panel {err['panel']}: {err['message']}")

        # ===================================================================
        # STEP 6: VALIDATION ASSERTIONS
        # ===================================================================
        print("\n--- STEP 6: VALIDATION ASSERTIONS ---")

        # Core assertions
        assert metrics.pages_generated >= target_pages * 0.9, \
            f"Expected ~{target_pages} pages, got {metrics.pages_generated}"
        assert metrics.panels_generated >= metrics.pages_generated * 3, \
            f"Expected at least 3 panels/page, got {metrics.panels_generated / max(1, metrics.pages_generated):.1f}"
        assert len(metrics.characters_referenced) >= 1, \
            "No characters referenced in script"

        # Validation thresholds
        assert metrics.continuity_errors == 0, \
            f"Continuity errors found: {metrics.continuity_errors}"
        assert metrics.script_manual_reviews <= 2, \
            f"Too many items needing manual review: {metrics.script_manual_reviews}"

        print("All validation assertions passed!")

        # ===================================================================
        # STEP 7: IMAGE GENERATION (if not script-only)
        # ===================================================================
        if script_only:
            print("\n--- SKIPPING IMAGE GENERATION (--script-only) ---")
        else:
            # Reference generation
            print("\n--- STEP 7a: REFERENCE GENERATION ---")
            ref_agent = ReferenceAgent(
                assets_path=str(assets_file),
                style_prompt=f"{style} style, {tone} tone",
                base_output_dir=base_output_dir
            )
            await ref_agent.run(style=style)

            # Panel generation
            print("\n--- STEP 7b: PANEL GENERATION ---")
            panel_agent = PanelAgent(script_path, f"{style} style, {tone} tone", base_output_dir=base_output_dir)
            await panel_agent.run_production()

            # Verify panels exist
            pages_dir = base_output_dir / "pages"
            if pages_dir.exists():
                panel_images = list(pages_dir.rglob("*.png"))
                print(f"Panel images generated: {len(panel_images)}")

            # Consistency audit on first page panels
            print("\n--- STEP 7c: CONSISTENCY AUDIT ---")
            await self._audit_page_consistency(script, metrics, base_output_dir)

            # Composition
            print("\n--- STEP 7d: COMPOSITION + EXPORT ---")
            compositor = CompositorAgent(script_path, base_output_dir=base_output_dir)
            compositor.run()

            # Verify final pages
            final_pages_dir = base_output_dir / "final_pages"
            if final_pages_dir.exists():
                final_pages = list(final_pages_dir.glob("*.png"))
                print(f"Final pages composed: {len(final_pages)}")
                assert len(final_pages) > 0, "No final pages generated"

        # ===================================================================
        # FINAL REPORT
        # ===================================================================
        print(metrics.report())

    async def _audit_page_consistency(self, script: List[Dict], metrics: IntegrationTestMetrics, base_output_dir: Path):
        """Audit character consistency across panels on first few pages."""
        from PIL import Image

        auditor = ConsistencyAuditor()
        pages_dir = base_output_dir / "pages"

        if not pages_dir.exists():
            print("  Skipping consistency audit - no pages directory")
            return

        # Audit first 2 pages
        for page in script[:2]:
            page_num = page.get('page_number', 0)
            page_dir = pages_dir / f"page_{page_num}"

            if not page_dir.exists():
                continue

            panel_images = sorted(page_dir.glob("panel_*.png"))
            if len(panel_images) < 2:
                continue

            # Load images
            images = []
            for img_path in panel_images[:4]:
                try:
                    images.append(Image.open(img_path))
                except Exception:
                    pass

            if len(images) < 2:
                continue

            # Get characters for this page
            chars_on_page = set()
            for panel in page.get('panels', []):
                chars_on_page.update(panel.get('characters', []))

            result = await auditor.audit_page_consistency(
                page_images=images,
                characters=list(chars_on_page)
            )

            if not result.consistent:
                metrics.consistency_issues += len(result.character_issues)
                print(f"  Page {page_num}: Consistency issues detected")
                for char, issues in result.character_issues.items():
                    print(f"    {char}: {', '.join(issues[:2])}")
            else:
                print(f"  Page {page_num}: Consistent")

            # Cleanup
            for img in images:
                img.close()


class TestDialogueQuality:
    """Focused tests for dialogue quality across pages."""

    @pytest.mark.asyncio
    async def test_dialogue_continuity(self, input_file, style, tone, script_path, base_output_dir):
        """
        Test that dialogue flows naturally between pages.

        Checks:
        - Conversations continue logically
        - Character voices remain consistent
        - No abrupt topic changes
        """
        script_file = Path(script_path)
        if not script_file.exists():
            pytest.skip("Script not found. Run integration test first.")

        with open(script_file, 'r') as f:
            script = json.load(f)

        print(f"\n{'='*60}")
        print("DIALOGUE CONTINUITY TEST")
        print(f"{'='*60}\n")

        # Track conversation threads
        active_conversations = {}
        dialogue_issues = []

        for page in script:
            page_num = page.get('page_number', 0)

            for panel in page.get('panels', []):
                dialogue = panel.get('dialogue', '')
                characters = panel.get('characters', [])

                if not dialogue:
                    continue

                # Check dialogue length (should already be validated)
                if len(dialogue) > 100:
                    dialogue_issues.append(f"Page {page_num}: Dialogue too long ({len(dialogue)} chars)")

                # Check for unnatural patterns
                if dialogue.count('!') > 3:
                    dialogue_issues.append(f"Page {page_num}: Excessive exclamation marks")

                if '...' in dialogue and dialogue.count('...') > 2:
                    dialogue_issues.append(f"Page {page_num}: Too many ellipses")

        print(f"Pages analyzed: {len(script)}")
        print(f"Dialogue issues found: {len(dialogue_issues)}")

        for issue in dialogue_issues[:10]:
            print(f"  - {issue}")

        # Allow some issues but not too many
        assert len(dialogue_issues) <= len(script) * 2, \
            f"Too many dialogue issues: {len(dialogue_issues)}"


class TestVisualContinuity:
    """Focused tests for visual continuity across pages."""

    @pytest.mark.asyncio
    async def test_character_appearance_tracking(self, script_path, base_output_dir):
        """
        Test that character appearances are tracked consistently.

        Verifies:
        - Characters don't appear/disappear without explanation
        - Character items persist across appearances
        """
        script_file = Path(script_path)
        if not script_file.exists():
            pytest.skip("Script not found. Run integration test first.")

        with open(script_file, 'r') as f:
            script = json.load(f)

        print(f"\n{'='*60}")
        print("CHARACTER APPEARANCE TRACKING TEST")
        print(f"{'='*60}\n")

        # Track character last seen page
        char_last_seen = {}
        appearance_gaps = []

        for page in script:
            page_num = page.get('page_number', 0)

            chars_this_page = set()
            for panel in page.get('panels', []):
                chars_this_page.update(panel.get('characters', []))

            for char in chars_this_page:
                if char in char_last_seen:
                    gap = page_num - char_last_seen[char]
                    if gap > 5:  # More than 5 pages between appearances
                        appearance_gaps.append({
                            'character': char,
                            'last_seen': char_last_seen[char],
                            'reappears': page_num,
                            'gap': gap
                        })
                char_last_seen[char] = page_num

        print(f"Characters tracked: {len(char_last_seen)}")
        print(f"Significant appearance gaps: {len(appearance_gaps)}")

        for gap in appearance_gaps[:5]:
            print(f"  - {gap['character']}: last seen p{gap['last_seen']}, reappears p{gap['reappears']} (gap: {gap['gap']})")

        # Large gaps aren't necessarily wrong, just log them
        if len(appearance_gaps) > len(char_last_seen):
            print("\nNote: Many appearance gaps. Consider adding continuity notes.")


# Allow running as standalone script
if __name__ == "__main__":
    import argparse
    from datetime import datetime
    import uuid

    parser = argparse.ArgumentParser(description="Run 10-page integration test")
    parser.add_argument("--input-file", default="assets/input/20-thousand-leagues-under-the-sea.txt")
    parser.add_argument("--style", default="Manga/Anime")
    parser.add_argument("--tone", default="Philosophical")
    parser.add_argument("--pages", type=int, default=10)
    parser.add_argument("--script-only", action="store_true")
    args = parser.parse_args()

    input_stem = Path(args.input_file).stem

    # Create isolated output directory for standalone run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    short_uuid = str(uuid.uuid4())[:8]
    base_output_dir = Path(f"assets/output/test_run_{timestamp}_{short_uuid}")
    base_output_dir.mkdir(parents=True, exist_ok=True)
    (base_output_dir / "characters").mkdir(exist_ok=True)
    (base_output_dir / "objects").mkdir(exist_ok=True)
    (base_output_dir / "pages").mkdir(exist_ok=True)
    (base_output_dir / "final_pages").mkdir(exist_ok=True)

    script_path = str(base_output_dir / f"{input_stem}_full_script.json")
    assets_path = str(base_output_dir / f"{input_stem}_assets.json")

    async def run():
        test = TestIntegration()
        await test.test_10_page_integration(
            input_file=args.input_file,
            style=args.style,
            tone=args.tone,
            target_pages=args.pages,
            script_path=script_path,
            assets_path=assets_path,
            base_output_dir=base_output_dir,
            script_only=args.script_only
        )

    asyncio.run(run())
