"""
Shared utilities for the scripting package.

Contains the ScriptingContext dataclass for passing state between passes
and the rate limiter initialization for parallel script generation.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from utils import RateLimiter
from config import config


# Rate limiter for the parallel script writing phase (lazy-initialized)
_scribe_limiter: Optional[RateLimiter] = None


def get_scribe_limiter() -> RateLimiter:
    """Get the scripting rate limiter with effective RPM based on config."""
    global _scribe_limiter
    if _scribe_limiter is None:
        effective_limits = config.get_effective_rate_limits()
        _scribe_limiter = RateLimiter(rpm_limit=effective_limits['scripting_rpm'])
    return _scribe_limiter


@dataclass
class ScriptingContext:
    """
    Shared context object passed between scripting passes.

    This dataclass holds all the state needed during script generation,
    including inputs, configuration, and intermediate outputs from each pass.
    """
    # Required inputs
    book_path: Path
    output_dir: Path
    style: str

    # Configuration
    context_constraints: str = ""
    target_pages: int = 10
    test_mode: bool = True

    # Cache state
    cache_name: Optional[str] = None
    full_text: Optional[str] = None

    # Pass outputs (populated as pipeline runs)
    beat_map: Optional[dict] = None
    adaptation_filter: Optional[dict] = None
    blueprint: Optional[list] = None
    character_arcs: Optional[dict] = None
    asset_manifest: Optional[dict] = None
    full_script: Optional[list] = None

    def get_book_stem(self) -> str:
        """Return the stem (filename without extension) of the book."""
        return self.book_path.stem
