"""
Test scripting agent JSON generation only (no image generation).

This test runs only the ScriptingAgent to generate:
- _full_script.json (or _test_page.json in test mode)
- _assets.json (character/object manifest)
- _beats.json (narrative beat analysis)
- _character_arcs.json (character development tracking)
- _adaptation.json (adaptation filter results)
- _validation.json (era/dialogue validation)

Useful for:
- Faster iteration when tweaking prompts
- Lower API costs (no image generation)
- Validating the 7-pass enrichment pipeline

Usage:
    # Default (test mode, 1 page)
    pytest tests/test_scripting_only.py -v

    # Custom input file
    pytest tests/test_scripting_only.py --input-file assets/input/alice.txt -v

    # Custom style and pages
    pytest tests/test_scripting_only.py --style "Lush Watercolor" --pages 3 -v

    # Full mode (no content reduction)
    pytest tests/test_scripting_only.py --full-mode --pages 10 -v
"""
import asyncio
import json
import sys
from pathlib import Path

import pytest

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents import ScriptingAgent


class TestScriptingOnly:
    """Test suite for ScriptingAgent JSON generation."""

    @pytest.mark.asyncio
    async def test_generate_script(self, input_file, style, tone, test_mode, target_pages, script_path, base_output_dir):
        """
        Test that ScriptingAgent generates valid script JSON.

        This runs the 7-pass enrichment pipeline:
        PASS 0: Global Context
        PASS 1: Beat Analysis
        PASS 1.5: Adaptation Filter
        PASS 2: Director Pass
        PASS 3: Character Deep Dive
        PASS 4: Asset Manifest
        PASS 5: Scriptwriter (parallel)
        PASS 6: Validation + Auto-Fix
        """
        print(f"\n{'='*60}")
        print("SCRIPTING-ONLY TEST")
        print(f"{'='*60}")
        print(f"Input:       {input_file}")
        print(f"Style:       {style}")
        print(f"Tone:        {tone}")
        print(f"Test Mode:   {test_mode}")
        print(f"Pages:       {target_pages}")
        print(f"Output Dir:  {base_output_dir}")
        print(f"Script:      {script_path}")
        print(f"{'='*60}\n")

        # Initialize and run scripting agent with isolated output directory
        scripter = ScriptingAgent(input_file, base_output_dir=base_output_dir)
        script = await scripter.generate_script(
            style=f"{style}, {tone}",
            test_mode=test_mode,
            target_page_override=target_pages if not test_mode else None
        )

        # Verify script was generated
        assert script is not None, "Script generation returned None"
        assert isinstance(script, list), f"Script should be a list, got {type(script)}"
        assert len(script) > 0, "Script should have at least one page"

        # Verify script structure
        for page in script:
            assert "page_number" in page, "Page missing page_number"
            assert "panels" in page, "Page missing panels"
            assert isinstance(page["panels"], list), "Panels should be a list"

            for panel in page["panels"]:
                assert "panel_id" in panel, "Panel missing panel_id"
                assert "visual_description" in panel, "Panel missing visual_description"

        # Verify output files exist in the isolated output directory
        input_stem = Path(input_file).stem

        expected_files = [
            f"{input_stem}_assets.json",
            f"{input_stem}_beats.json",
            f"{input_stem}_character_arcs.json",
        ]

        # Check script file
        script_file = Path(script_path)
        assert script_file.exists(), f"Script file not found: {script_path}"

        # Check supporting files in the isolated output directory
        for filename in expected_files:
            filepath = base_output_dir / filename
            # These files are generated in different passes, log status
            if filepath.exists():
                print(f"  [OK] {filename}")
            else:
                print(f"  [--] {filename} (may be generated in later passes)")

        print(f"\nScript generated with {len(script)} page(s)")
        total_panels = sum(len(p["panels"]) for p in script)
        print(f"Total panels: {total_panels}")

    @pytest.mark.asyncio
    async def test_script_json_validity(self, input_file, style, tone, test_mode, script_path, base_output_dir):
        """Test that generated script JSON is valid and parseable."""
        # First generate the script with isolated output directory
        scripter = ScriptingAgent(input_file, base_output_dir=base_output_dir)
        await scripter.generate_script(
            style=f"{style}, {tone}",
            test_mode=test_mode
        )

        # Read and validate JSON structure
        script_file = Path(script_path)
        assert script_file.exists(), f"Script file not found: {script_path}"

        with open(script_file, 'r') as f:
            script_data = json.load(f)

        # Validate it's a list of pages
        assert isinstance(script_data, list), "Script should be a list"

        # Validate each page has required fields
        required_page_fields = {"page_number", "panels"}
        required_panel_fields = {"panel_id", "visual_description"}

        for page in script_data:
            missing_page = required_page_fields - set(page.keys())
            assert not missing_page, f"Page missing fields: {missing_page}"

            for panel in page["panels"]:
                missing_panel = required_panel_fields - set(panel.keys())
                assert not missing_panel, f"Panel missing fields: {missing_panel}"


# Allow running as standalone script
if __name__ == "__main__":
    import argparse
    from datetime import datetime
    import uuid

    parser = argparse.ArgumentParser(description="Run scripting-only test")
    parser.add_argument("--input-file", default="assets/input/20-thousand-leagues-under-the-sea.txt")
    parser.add_argument("--style", default="Manga/Anime")
    parser.add_argument("--tone", default="Philosophical")
    parser.add_argument("--pages", type=int, default=1)
    parser.add_argument("--full-mode", action="store_true")
    args = parser.parse_args()

    test_mode = not args.full_mode
    input_stem = Path(args.input_file).stem
    suffix = "_test_page" if test_mode else "_full_script"

    # Create isolated output directory for standalone run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    short_uuid = str(uuid.uuid4())[:8]
    base_output_dir = Path(f"assets/output/test_run_{timestamp}_{short_uuid}")
    base_output_dir.mkdir(parents=True, exist_ok=True)
    (base_output_dir / "characters").mkdir(exist_ok=True)
    (base_output_dir / "objects").mkdir(exist_ok=True)
    (base_output_dir / "pages").mkdir(exist_ok=True)
    (base_output_dir / "final_pages").mkdir(exist_ok=True)

    script_path = str(base_output_dir / f"{input_stem}{suffix}.json")

    async def run():
        test = TestScriptingOnly()
        await test.test_generate_script(
            args.input_file, args.style, args.tone,
            test_mode, args.pages, script_path, base_output_dir
        )

    asyncio.run(run())
