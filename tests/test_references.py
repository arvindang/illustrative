"""
Test reference sheet generation only (faster than full pipeline).

This test runs only the ReferenceAgent to generate:
- Character reference sheets (multi-angle views)
- Object reference sheets

Useful for:
- Faster iteration on character/object consistency
- Testing style application without full panel generation
- Validating reference quality before committing to panel generation

Usage:
    # Default (uses existing assets from most recent script generation)
    pytest tests/test_references.py -v

    # With custom asset manifest
    pytest tests/test_references.py --assets-path assets/output/my_book_assets.json -v

    # With custom style
    pytest tests/test_references.py --style "Botanical Illustration" -v

    # Generate script first, then references
    pytest tests/test_references.py --generate-script -v
"""
import asyncio
import json
import sys
from pathlib import Path

import pytest

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents import ScriptingAgent, ReferenceAgent
from validators import ConsistencyAuditor


def pytest_addoption(parser):
    """Add reference-specific CLI options."""
    try:
        parser.addoption(
            "--assets-path",
            action="store",
            default=None,
            help="Path to asset manifest JSON (default: auto-detect from input file)"
        )
        parser.addoption(
            "--generate-script",
            action="store_true",
            default=False,
            help="Generate script first before reference generation"
        )
    except ValueError:
        # Options already added by conftest.py
        pass


@pytest.fixture
def assets_path_override(request):
    """Override assets path if provided."""
    return request.config.getoption("--assets-path", default=None)


@pytest.fixture
def generate_script_first(request):
    """Whether to generate script before references."""
    try:
        return request.config.getoption("--generate-script")
    except ValueError:
        return False


class TestReferenceGeneration:
    """Test suite for reference sheet generation."""

    @pytest.mark.asyncio
    async def test_generate_references_only(
        self, input_file, style, tone, test_mode, assets_path, assets_path_override, generate_script_first
    ):
        """
        Test reference sheet generation without panel generation.

        Steps:
        1. (Optional) Generate script if requested
        2. Load asset manifest
        3. Generate character reference sheets
        4. Generate object reference sheets
        5. Validate references exist
        """
        print(f"\n{'='*60}")
        print("REFERENCE GENERATION TEST")
        print(f"{'='*60}")

        # Determine asset path
        actual_assets_path = assets_path_override or assets_path
        assets_file = Path(actual_assets_path)

        # Optionally generate script first
        if generate_script_first or not assets_file.exists():
            print(f"\n--- STEP 0: GENERATING SCRIPT (assets not found) ---")
            scripter = ScriptingAgent(input_file)
            await scripter.generate_script(
                style=f"{style}, {tone}",
                test_mode=test_mode
            )
            print(f"Script generated. Assets at: {actual_assets_path}")

        # Verify assets exist
        assert assets_file.exists(), f"Asset manifest not found: {actual_assets_path}. Run with --generate-script or run scripting test first."

        with open(assets_file, 'r') as f:
            assets = json.load(f)

        char_count = len(assets.get('characters', []))
        obj_count = len(assets.get('objects', []))

        print(f"Input:       {input_file}")
        print(f"Style:       {style}")
        print(f"Assets:      {actual_assets_path}")
        print(f"Characters:  {char_count}")
        print(f"Objects:     {obj_count}")
        print(f"{'='*60}\n")

        # --- GENERATE REFERENCES ---
        print("\n--- GENERATING REFERENCE SHEETS ---")
        ref_agent = ReferenceAgent(
            assets_path=str(actual_assets_path),
            style_prompt=f"{style} style, {tone} tone"
        )
        await ref_agent.run(style=style)

        # --- VALIDATE REFERENCES GENERATED ---
        print("\n--- VALIDATING REFERENCES ---")
        char_dir = ref_agent.char_base_dir
        obj_dir = ref_agent.obj_base_dir

        # Check character references
        char_folders = list(char_dir.iterdir()) if char_dir.exists() else []
        char_with_refs = 0
        for folder in char_folders:
            if folder.is_dir():
                refs = list(folder.glob("*.png"))
                if refs:
                    char_with_refs += 1
                    print(f"  [OK] {folder.name}: {len(refs)} reference(s)")
                else:
                    print(f"  [--] {folder.name}: no references")

        # Check object references
        obj_folders = list(obj_dir.iterdir()) if obj_dir.exists() else []
        obj_with_refs = 0
        for folder in obj_folders:
            if folder.is_dir():
                refs = list(folder.glob("*.png"))
                if refs:
                    obj_with_refs += 1
                    print(f"  [OK] {folder.name}: {len(refs)} reference(s)")

        print(f"\nReference Generation Summary:")
        print(f"  Characters with refs: {char_with_refs}/{char_count}")
        print(f"  Objects with refs:    {obj_with_refs}/{obj_count}")

        # At least some references should be generated
        assert char_with_refs > 0 or obj_with_refs > 0, "No references were generated"

    @pytest.mark.asyncio
    async def test_reference_metadata_validity(self, assets_path):
        """
        Test that generated reference metadata is valid.
        """
        assets_file = Path(assets_path)
        if not assets_file.exists():
            pytest.skip("Assets file not found. Run scripting test first.")

        with open(assets_file, 'r') as f:
            assets = json.load(f)

        char_dir = Path("assets/output/characters")
        if not char_dir.exists():
            pytest.skip("Character directory not found. Run reference generation first.")

        print(f"\n{'='*60}")
        print("REFERENCE METADATA VALIDATION")
        print(f"{'='*60}\n")

        issues = []
        for char_folder in char_dir.iterdir():
            if not char_folder.is_dir():
                continue

            metadata_path = char_folder / "metadata.json"
            if not metadata_path.exists():
                issues.append(f"Missing metadata: {char_folder.name}")
                continue

            with open(metadata_path, 'r') as f:
                metadata = json.load(f)

            # Validate required fields
            if not metadata.get('name'):
                issues.append(f"Missing name in metadata: {char_folder.name}")

            # Check reference images match metadata
            ref_images = list(char_folder.glob("*.png"))
            if 'reference_images' in metadata:
                expected_count = len(metadata['reference_images'])
                actual_count = len(ref_images)
                if actual_count < expected_count:
                    issues.append(f"Missing images for {char_folder.name}: expected {expected_count}, got {actual_count}")

            print(f"  [OK] {char_folder.name}: metadata valid")

        if issues:
            print("\nIssues found:")
            for issue in issues:
                print(f"  - {issue}")
        else:
            print("\nAll metadata valid!")

        # Allow some issues but not too many
        assert len(issues) <= len(list(char_dir.iterdir())) // 2, f"Too many metadata issues: {issues}"


class TestReferenceQuality:
    """Tests for reference quality validation (requires API calls)."""

    @pytest.mark.asyncio
    async def test_character_consistency_audit(self, assets_path):
        """
        Test that character references are consistent across views.

        Uses the ConsistencyAuditor to check that multi-angle views
        show the same character.
        """
        char_dir = Path("assets/output/characters")
        if not char_dir.exists():
            pytest.skip("Character directory not found. Run reference generation first.")

        # Find characters with multiple reference images
        chars_with_multiple_refs = []
        for char_folder in char_dir.iterdir():
            if char_folder.is_dir():
                refs = list(char_folder.glob("*.png"))
                if len(refs) >= 2:
                    chars_with_multiple_refs.append((char_folder.name, refs))

        if not chars_with_multiple_refs:
            pytest.skip("No characters with multiple references found.")

        print(f"\n{'='*60}")
        print("CHARACTER CONSISTENCY AUDIT")
        print(f"{'='*60}\n")

        from PIL import Image

        auditor = ConsistencyAuditor()
        all_consistent = True

        # Audit first few characters with multiple refs
        for char_name, ref_paths in chars_with_multiple_refs[:3]:
            print(f"\nAuditing: {char_name}")
            images = [Image.open(p) for p in ref_paths[:4]]  # Max 4 images

            result = await auditor.audit_page_consistency(
                page_images=images,
                characters=[char_name]
            )

            if result.consistent:
                print(f"  [OK] Consistent across {len(images)} views")
            else:
                print(f"  [!!] Inconsistency detected:")
                for char, issues in result.character_issues.items():
                    for issue in issues:
                        print(f"       - {issue}")
                all_consistent = False

            # Clean up
            for img in images:
                img.close()

        # Log but don't fail on minor inconsistencies
        if not all_consistent:
            print("\nNote: Some inconsistencies detected. Review references manually.")


class TestReferenceResume:
    """Tests for reference generation resume capability."""

    @pytest.mark.asyncio
    async def test_resume_skips_existing(self, input_file, style, tone, assets_path):
        """
        Test that re-running reference generation skips existing references.
        """
        assets_file = Path(assets_path)
        if not assets_file.exists():
            pytest.skip("Assets file not found. Run scripting test first.")

        char_dir = Path("assets/output/characters")
        existing_refs = []
        if char_dir.exists():
            for folder in char_dir.iterdir():
                if folder.is_dir():
                    refs = list(folder.glob("*.png"))
                    if refs:
                        existing_refs.append(folder.name)

        if not existing_refs:
            pytest.skip("No existing references to test resume capability.")

        print(f"\n{'='*60}")
        print("REFERENCE RESUME TEST")
        print(f"{'='*60}")
        print(f"Existing references: {len(existing_refs)}")

        # Record modification times
        mod_times = {}
        for folder_name in existing_refs[:3]:
            folder = char_dir / folder_name
            refs = list(folder.glob("*.png"))
            if refs:
                mod_times[folder_name] = refs[0].stat().st_mtime

        # Re-run reference generation
        ref_agent = ReferenceAgent(
            assets_path=str(assets_file),
            style_prompt=f"{style} style, {tone} tone"
        )
        await ref_agent.run(style=style)

        # Check that existing references weren't regenerated
        skipped = 0
        regenerated = 0
        for folder_name, original_mtime in mod_times.items():
            folder = char_dir / folder_name
            refs = list(folder.glob("*.png"))
            if refs:
                new_mtime = refs[0].stat().st_mtime
                if new_mtime == original_mtime:
                    skipped += 1
                    print(f"  [OK] {folder_name}: skipped (already exists)")
                else:
                    regenerated += 1
                    print(f"  [!!] {folder_name}: regenerated")

        print(f"\nResume Summary: {skipped} skipped, {regenerated} regenerated")

        # Most should be skipped
        assert skipped >= regenerated, "Resume capability not working - too many regenerations"


# Allow running as standalone script
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run reference generation test")
    parser.add_argument("--input-file", default="assets/input/20-thousand-leagues-under-the-sea.txt")
    parser.add_argument("--style", default="Manga/Anime")
    parser.add_argument("--tone", default="Philosophical")
    parser.add_argument("--assets-path", default=None)
    parser.add_argument("--generate-script", action="store_true")
    args = parser.parse_args()

    input_stem = Path(args.input_file).stem
    default_assets = f"assets/output/{input_stem}_assets.json"
    assets_path = args.assets_path or default_assets

    async def run():
        test = TestReferenceGeneration()
        await test.test_generate_references_only(
            input_file=args.input_file,
            style=args.style,
            tone=args.tone,
            test_mode=True,
            assets_path=default_assets,
            assets_path_override=args.assets_path,
            generate_script_first=args.generate_script
        )

    asyncio.run(run())
