"""
Test complete pipeline: Scripting -> Illustration -> Composition.

This test runs all three agents:
1. ScriptingAgent: Generates script + asset manifest (7-pass enrichment)
2. IllustratorAgent: Generates character references + panel images
3. CompositorAgent: Composes pages + exports to final_pages/

Usage:
    # Default (test mode, 1 page)
    pytest tests/test_full_pipeline.py -v

    # Custom configuration
    pytest tests/test_full_pipeline.py --style "Lush Watercolor" --pages 5 -v

    # Full production mode
    pytest tests/test_full_pipeline.py --full-mode --pages 10 -v

Note: This test makes API calls for image generation and will incur costs.
Use test_scripting_only.py for cheaper iteration on script generation.
"""
import asyncio
import sys
from pathlib import Path

import pytest

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents import ScriptingAgent, IllustratorAgent, CompositorAgent


class TestFullPipeline:
    """Test suite for complete 3-step pipeline."""

    @pytest.mark.asyncio
    async def test_full_pipeline(self, input_file, style, tone, test_mode, target_pages, script_path):
        """
        Test complete pipeline from text to final composed pages.

        Steps:
        1. ScriptingAgent: Script + asset manifest
        2. IllustratorAgent: Reference sheets + panel images
        3. CompositorAgent: Page composition + export
        """
        print(f"\n{'='*60}")
        print("FULL PIPELINE TEST")
        print(f"{'='*60}")
        print(f"Input:       {input_file}")
        print(f"Style:       {style}")
        print(f"Tone:        {tone}")
        print(f"Test Mode:   {test_mode}")
        print(f"Pages:       {target_pages}")
        print(f"{'='*60}\n")

        # --- STEP 1: SCRIPTING ---
        print("\n--- STEP 1: SCRIPTING + ASSET ANALYSIS ---")
        scripter = ScriptingAgent(input_file)
        script = await scripter.generate_script(
            style=f"{style}, {tone}",
            test_mode=test_mode,
            target_page_override=target_pages if not test_mode else None
        )

        assert script is not None, "Script generation failed"
        assert len(script) > 0, "Script should have at least one page"
        print(f"Script generated: {len(script)} page(s)")

        # --- STEP 2: ILLUSTRATION ---
        print("\n--- STEP 2: ILLUSTRATION ---")
        illustrator = IllustratorAgent(script_path, f"{style} style, {tone} tone")

        # Generate reference sheets for characters/objects
        await illustrator.generate_all_references(style=style)

        # Generate panel images
        await illustrator.run_production()

        # Verify panel images were generated
        pages_dir = Path("assets/output/pages")
        if pages_dir.exists():
            panel_images = list(pages_dir.glob("*.png"))
            print(f"Panel images generated: {len(panel_images)}")
            assert len(panel_images) > 0, "No panel images were generated"

        # --- STEP 3: COMPOSITION ---
        print("\n--- STEP 3: COMPOSITION + EXPORT ---")
        compositor = CompositorAgent(script_path)
        compositor.run()

        # Verify final pages were generated
        final_pages_dir = Path("assets/output/final_pages")
        assert final_pages_dir.exists(), "Final pages directory not created"

        final_pages = list(final_pages_dir.glob("*.png"))
        print(f"Final pages composed: {len(final_pages)}")
        assert len(final_pages) > 0, "No final pages were generated"

        print(f"\nPipeline complete! Check: assets/output/final_pages/")

    @pytest.mark.asyncio
    async def test_resume_from_script(self, script_path, style, tone):
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


# Allow running as standalone script
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run full pipeline test")
    parser.add_argument("--input-file", default="assets/input/20-thousand-leagues-under-the-sea.txt")
    parser.add_argument("--style", default="Manga/Anime")
    parser.add_argument("--tone", default="Philosophical")
    parser.add_argument("--pages", type=int, default=1)
    parser.add_argument("--full-mode", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Resume from existing script")
    args = parser.parse_args()

    test_mode = not args.full_mode
    input_stem = Path(args.input_file).stem
    suffix = "_test_page" if test_mode else "_full_script"
    script_path = f"assets/output/{input_stem}{suffix}.json"

    async def run():
        test = TestFullPipeline()
        if args.resume:
            await test.test_resume_from_script(script_path, args.style, args.tone)
        else:
            await test.test_full_pipeline(
                args.input_file, args.style, args.tone,
                test_mode, args.pages, script_path
            )

    asyncio.run(run())
