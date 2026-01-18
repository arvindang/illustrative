"""
Shared pytest fixtures and CLI configuration for Illustrative AI tests.

Usage:
    # Run all tests with defaults
    pytest tests/

    # Run scripting-only test with custom input
    pytest tests/test_scripting_only.py --input-file assets/input/my_book.txt

    # Run full pipeline with custom style and pages
    pytest tests/test_full_pipeline.py --style "Manga/Anime" --pages 5

Requirements:
    pip install pytest pytest-asyncio

Output Isolation:
    Each test session creates a unique output directory under assets/output/
    with format: test_run_YYYYMMDD_HHMMSS_{short_uuid}/
    This prevents stale file references and allows easy comparison between runs.
"""
import pytest
import pytest_asyncio
from pathlib import Path
from datetime import datetime
import uuid

# Configure pytest-asyncio to use auto mode
pytest_plugins = ('pytest_asyncio',)


def pytest_configure(config):
    """Configure pytest-asyncio mode."""
    config.addinivalue_line(
        "markers", "asyncio: mark test as async"
    )


def pytest_addoption(parser):
    """Add custom CLI options for test configuration."""
    parser.addoption(
        "--input-file",
        action="store",
        default="assets/input/20-thousand-leagues-under-the-sea.txt",
        help="Path to input text file (default: 20-thousand-leagues-under-the-sea.txt)"
    )
    parser.addoption(
        "--style",
        action="store",
        default="Manga/Anime",
        help="Visual style for generation (default: Manga/Anime)"
    )
    parser.addoption(
        "--tone",
        action="store",
        default="Philosophical",
        help="Narrative tone (default: Philosophical)"
    )
    parser.addoption(
        "--pages",
        action="store",
        type=int,
        default=1,
        help="Target number of pages (default: 1)"
    )
    parser.addoption(
        "--test-mode",
        action="store_true",
        default=True,
        help="Run in test mode with reduced content (default: True)"
    )
    parser.addoption(
        "--full-mode",
        action="store_true",
        default=False,
        help="Run in full production mode (disables test mode)"
    )
    parser.addoption(
        "--skip-validation",
        action="store_true",
        default=False,
        help="Skip validation steps for faster testing (smoke test)"
    )


@pytest.fixture
def input_file(request):
    """Input text file path."""
    return request.config.getoption("--input-file")


@pytest.fixture
def style(request):
    """Visual style for generation."""
    return request.config.getoption("--style")


@pytest.fixture
def tone(request):
    """Narrative tone."""
    return request.config.getoption("--tone")


@pytest.fixture
def target_pages(request):
    """Target number of pages to generate."""
    return request.config.getoption("--pages")


@pytest.fixture
def test_mode(request):
    """Whether to run in test mode (reduced content)."""
    if request.config.getoption("--full-mode"):
        return False
    return request.config.getoption("--test-mode")


@pytest.fixture(scope="session")
def base_output_dir():
    """
    Create a unique output directory for this test session.

    Format: assets/output/test_run_YYYYMMDD_HHMMSS_{short_uuid}/

    This ensures:
    - Each test run is isolated from previous runs
    - No stale file references
    - Easy comparison between different test runs
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    short_uuid = str(uuid.uuid4())[:8]
    output_dir = Path(f"assets/output/test_run_{timestamp}_{short_uuid}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create subdirectories that agents expect
    (output_dir / "characters").mkdir(exist_ok=True)
    (output_dir / "objects").mkdir(exist_ok=True)
    (output_dir / "pages").mkdir(exist_ok=True)
    (output_dir / "final_pages").mkdir(exist_ok=True)

    print(f"\n📁 Test output directory: {output_dir}")
    return output_dir


@pytest.fixture
def script_path(input_file, test_mode, base_output_dir):
    """Derived script output path based on input file and mode."""
    input_stem = Path(input_file).stem
    suffix = "_test_page" if test_mode else "_full_script"
    return str(base_output_dir / f"{input_stem}{suffix}.json")


@pytest.fixture
def assets_path(input_file, base_output_dir):
    """Derived assets manifest path based on input file."""
    input_stem = Path(input_file).stem
    return str(base_output_dir / f"{input_stem}_assets.json")
