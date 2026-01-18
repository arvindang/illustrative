import asyncio
import random
import functools
import json
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Callable
from google.api_core import exceptions
from google import genai

# ============================================================================
# GEMINI CLIENT (Centralized - supports both Vertex AI and API key modes)
# ============================================================================

_client = None
_client_config = None


def get_client():
    """
    Returns a Gemini client, supporting both Vertex AI and API key modes.

    Configuration is read from config.py:
    - use_vertex_ai: If True, uses Vertex AI (requires GCP project + ADC)
    - gcp_project: GCP project ID for Vertex AI
    - gcp_location: GCP region (default: us-central1)
    - gemini_api_key: API key for Google AI Studio (non-Vertex mode)

    The client is cached and reused across calls. If config changes,
    a new client is created.
    """
    global _client, _client_config

    # Import config here to avoid circular import
    from config import config

    # Determine current config state
    current_config = (config.use_vertex_ai, config.gcp_project, config.gemini_api_key)

    if _client is None or _client_config != current_config:
        if config.use_vertex_ai:
            _client = genai.Client(
                vertexai=True,
                project=config.gcp_project,
                location=config.gcp_location
            )
        else:
            _client = genai.Client(api_key=config.gemini_api_key)
        _client_config = current_config
    return _client


def reset_client():
    """
    Reset the cached client. Useful when config changes at runtime.
    """
    global _client, _client_config
    _client = None
    _client_config = None

class APITimeoutError(Exception):
    """Custom timeout error with context for API operations."""
    def __init__(self, message: str, operation: str = None, timeout: int = None):
        self.operation = operation
        self.timeout = timeout
        super().__init__(message)


async def with_timeout(coro, timeout_seconds: int = 120, context: str = ""):
    """
    Wrap a coroutine with a timeout.

    Args:
        coro: The coroutine to execute
        timeout_seconds: Maximum seconds to wait
        context: Optional context string for error message

    Returns:
        The coroutine result

    Raises:
        APITimeoutError: If the coroutine doesn't complete in time
    """
    try:
        return await asyncio.wait_for(coro, timeout=timeout_seconds)
    except asyncio.TimeoutError:
        msg = f"Operation timed out after {timeout_seconds}s"
        if context:
            msg += f": {context}"
        raise APITimeoutError(msg, operation=context, timeout=timeout_seconds)


def retry_with_backoff(max_retries=5, initial_delay=2, max_delay=60, timeout_seconds=180):
    """
    Decorator for retrying async functions with exponential backoff, jitter, and timeout.
    Specifically targets 429 (ResourceExhausted) and 500/503 (Server Error).

    Args:
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay between retries in seconds
        max_delay: Maximum delay between retries
        timeout_seconds: Timeout for each individual attempt (default 180s = 3 minutes)
    """
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            delay = initial_delay
            last_exception = None

            for attempt in range(max_retries + 1):
                try:
                    # Wrap each attempt with timeout
                    return await asyncio.wait_for(
                        func(*args, **kwargs),
                        timeout=timeout_seconds
                    )
                except asyncio.TimeoutError:
                    last_exception = APITimeoutError(
                        f"{func.__name__} timed out after {timeout_seconds}s on attempt {attempt + 1}",
                        operation=func.__name__,
                        timeout=timeout_seconds
                    )
                    if attempt < max_retries:
                        print(f"⚠️ Timeout (Attempt {attempt+1}/{max_retries+1}), retrying...")
                        await asyncio.sleep(delay)
                        delay *= 2
                    else:
                        raise last_exception
                except Exception as e:
                    last_exception = e

                    # Check if the error is a rate limit or server error
                    # The google-genai SDK wraps these, but we check for common indicators
                    error_msg = str(e).lower()
                    is_rate_limit = "429" in error_msg or "resource_exhausted" in error_msg
                    is_server_error = "500" in error_msg or "503" in error_msg or "internal" in error_msg

                    if attempt < max_retries and (is_rate_limit or is_server_error):
                        # Calculate delay with jitter
                        jitter = random.uniform(0, 0.1) * delay
                        sleep_time = min(delay + jitter, max_delay)

                        print(f"⚠️ API Error (Attempt {attempt+1}/{max_retries+1}): {e}")
                        print(f"⏳ Retrying in {sleep_time:.2f} seconds...")

                        await asyncio.sleep(sleep_time)
                        delay *= 2 # Exponential backoff
                    else:
                        # If it's not a retryable error or we've hit max retries, raise it
                        raise last_exception
            return await func(*args, **kwargs)
        return wrapper
    return decorator


def calculate_dynamic_timeout(target_pages: int, base_timeout: int = 120, per_page_seconds: int = 3) -> int:
    """
    Calculate appropriate timeout based on job size.

    For blueprint generation, larger page counts require more processing time.

    Args:
        target_pages: Number of pages being generated
        base_timeout: Minimum timeout in seconds (default 120s)
        per_page_seconds: Additional seconds per page (default 3s)

    Returns:
        Calculated timeout in seconds (capped at 600s)
    """
    calculated = base_timeout + (target_pages * per_page_seconds)
    return min(600, calculated)  # Cap at 10 minutes max


async def retry_api_call(
    func,
    *args,
    max_retries: int = 5,
    initial_delay: int = 2,
    max_delay: int = 60,
    timeout_seconds: int = 180,
    **kwargs
):
    """
    Retry an async API call with exponential backoff and configurable timeout.

    Unlike the decorator, this allows dynamic timeout per call.

    Args:
        func: Async function to call
        *args: Positional arguments for func
        max_retries: Maximum retry attempts
        initial_delay: Initial backoff delay in seconds
        max_delay: Maximum backoff delay
        timeout_seconds: Timeout for each attempt
        **kwargs: Keyword arguments for func

    Returns:
        Result from successful func call

    Raises:
        APITimeoutError: If all attempts timeout
        Exception: If non-retryable error occurs
    """
    delay = initial_delay
    last_exception = None

    for attempt in range(max_retries + 1):
        try:
            return await asyncio.wait_for(
                func(*args, **kwargs),
                timeout=timeout_seconds
            )
        except asyncio.TimeoutError:
            last_exception = APITimeoutError(
                f"{func.__name__} timed out after {timeout_seconds}s on attempt {attempt + 1}",
                operation=func.__name__,
                timeout=timeout_seconds
            )
            if attempt < max_retries:
                print(f"⚠️ Timeout (Attempt {attempt+1}/{max_retries+1}), retrying...")
                await asyncio.sleep(delay)
                delay = min(delay * 2, max_delay)
            else:
                raise last_exception
        except Exception as e:
            last_exception = e
            error_msg = str(e).lower()
            is_rate_limit = "429" in error_msg or "resource_exhausted" in error_msg
            is_server_error = "500" in error_msg or "503" in error_msg or "internal" in error_msg

            if attempt < max_retries and (is_rate_limit or is_server_error):
                jitter = random.uniform(0, 0.1) * delay
                sleep_time = min(delay + jitter, max_delay)
                print(f"⚠️ API Error (Attempt {attempt+1}/{max_retries+1}): {e}")
                print(f"⏳ Retrying in {sleep_time:.2f} seconds...")
                await asyncio.sleep(sleep_time)
                delay = min(delay * 2, max_delay)
            else:
                raise last_exception

    return await func(*args, **kwargs)


class RateLimiter:
    """
    A simple semaphore-based rate limiter to prevent exceeding RPM limits.
    """
    def __init__(self, rpm_limit=15):
        self.semaphore = None
        self.rpm_limit = rpm_limit

    async def __aenter__(self):
        if self.semaphore is None:
            self.semaphore = asyncio.Semaphore(self.rpm_limit)
        await self.semaphore.acquire()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        # We hold the semaphore for a bit to ensure we don't burst too fast
        # (60 seconds / RPM) gives a rough spacing
        spacing = 60.0 / self.rpm_limit
        await asyncio.sleep(spacing)
        self.semaphore.release()


@dataclass
class TokenRecord:
    """Record of tokens used at a specific timestamp."""
    timestamp: float
    tokens: int


class TPMRateLimiter:
    """
    Token-per-minute aware rate limiter using a sliding window approach.

    Features:
    - Pre-call token estimation to prevent exceeding limits
    - Post-call token tracking from API response metadata
    - Sliding 60-second window for accurate TPM tracking
    - Configurable safety margin (e.g., 85% of limit)
    - Thread-safe async implementation
    """

    def __init__(
        self,
        tpm_limit: int = 1_000_000,
        safety_margin: float = 0.85,
        window_seconds: float = 60.0
    ):
        self.tpm_limit = tpm_limit
        self.effective_limit = int(tpm_limit * safety_margin)
        self.window_seconds = window_seconds

        # Sliding window storage
        self._token_records: deque = deque()
        self._lock = asyncio.Lock()

        # Current window usage
        self._current_tokens = 0

    def _prune_old_records(self) -> None:
        """Remove records older than the sliding window."""
        cutoff = time.time() - self.window_seconds
        while self._token_records and self._token_records[0].timestamp < cutoff:
            old_record = self._token_records.popleft()
            self._current_tokens -= old_record.tokens

    def get_current_usage(self) -> int:
        """Get current token usage in the sliding window."""
        self._prune_old_records()
        return self._current_tokens

    def get_available_tokens(self) -> int:
        """Get available tokens before hitting the limit."""
        return max(0, self.effective_limit - self.get_current_usage())

    async def wait_for_capacity(self, estimated_tokens: int) -> float:
        """
        Wait until there's enough capacity for the estimated tokens.
        Returns the number of seconds waited.
        """
        wait_time = 0.0

        async with self._lock:
            while True:
                self._prune_old_records()
                available = self.effective_limit - self._current_tokens

                if available >= estimated_tokens:
                    return wait_time

                # Calculate minimum wait time
                if self._token_records:
                    oldest = self._token_records[0]
                    time_until_free = (oldest.timestamp + self.window_seconds) - time.time()
                    sleep_time = max(0.1, time_until_free)
                else:
                    sleep_time = 1.0

                print(f"   ⏳ TPM limit: waiting {sleep_time:.1f}s ({self._current_tokens:,}/{self.effective_limit:,} tokens used)")
                await asyncio.sleep(sleep_time)
                wait_time += sleep_time

    def record_usage(self, tokens: int) -> None:
        """Record token usage after an API call."""
        record = TokenRecord(timestamp=time.time(), tokens=tokens)
        self._token_records.append(record)
        self._current_tokens += tokens

    async def acquire(self, estimated_tokens: int) -> None:
        """
        Acquire capacity for an API call.
        Waits if necessary, then pre-records the estimated usage.
        """
        await self.wait_for_capacity(estimated_tokens)
        self.record_usage(estimated_tokens)

    def update_actual_usage(self, estimated: int, actual: int) -> None:
        """
        Correct the recorded usage with actual token count from response.
        Called after receiving API response with usage_metadata.
        """
        difference = actual - estimated
        if difference != 0:
            self._current_tokens += difference
            # Add a correction record if we underestimated
            if difference > 0:
                self._token_records.append(TokenRecord(time.time(), difference))

    def acquire_sync(self, estimated_tokens: int) -> None:
        """
        Synchronous version of acquire for non-async contexts.
        Records usage without waiting (suitable for low-volume sync calls).
        """
        self._prune_old_records()
        available = self.effective_limit - self._current_tokens

        if available < estimated_tokens:
            # In sync context, we can't wait - just log a warning
            print(f"   ⚠️ TPM limit approaching: {self._current_tokens:,}/{self.effective_limit:,} tokens")

        self.record_usage(estimated_tokens)


# Global TPM limiter (singleton)
_global_tpm_limiter: Optional[TPMRateLimiter] = None


def get_tpm_limiter() -> TPMRateLimiter:
    """Get or create the global TPM limiter using effective rate limits."""
    global _global_tpm_limiter
    if _global_tpm_limiter is None:
        from config import config
        # Use effective rate limits (adjusts for Vertex AI vs AI Studio)
        effective_limits = config.get_effective_rate_limits()
        tpm_limit = effective_limits.get('tpm_limit', 1_000_000)
        tpm_safety_margin = getattr(config, 'tpm_safety_margin', 0.85)
        _global_tpm_limiter = TPMRateLimiter(
            tpm_limit=tpm_limit,
            safety_margin=tpm_safety_margin
        )
    return _global_tpm_limiter


def reset_tpm_limiter():
    """Reset the global TPM limiter. Call when config changes."""
    global _global_tpm_limiter
    _global_tpm_limiter = None


def estimate_tokens_for_text(prompt: str) -> int:
    """
    Estimate tokens for a text generation request.

    Uses character-based estimation:
    - ~4 characters per token for English text
    - Add output buffer (typically 20% of input or min 1000)
    """
    input_estimate = len(prompt) // 4
    output_estimate = max(1000, input_estimate // 5)
    return input_estimate + output_estimate


def estimate_tokens_for_image(prompt: str, num_reference_images: int = 0) -> int:
    """
    Estimate tokens for image generation.

    Image models have different token profiles:
    - Prompt: ~len(prompt)//4
    - Each reference image: ~1500 tokens (encoded in multimodal)
    - Output: ~500 tokens (mostly metadata)
    """
    prompt_tokens = len(prompt) // 4
    image_tokens = num_reference_images * 1500
    output_tokens = 500
    return prompt_tokens + image_tokens + output_tokens


def estimate_tokens_for_cache(content: str) -> int:
    """
    Estimate tokens for context cache creation.
    The entire book is tokenized and stored.
    """
    return len(content) // 4 + 1000


def extract_token_usage(response) -> tuple:
    """
    Extract token usage from Gemini API response.

    Returns:
        Tuple of (input_tokens, output_tokens)
    """
    if not hasattr(response, 'usage_metadata') or response.usage_metadata is None:
        return (0, 0)

    metadata = response.usage_metadata
    input_tokens = getattr(metadata, 'prompt_token_count', 0) or 0
    output_tokens = getattr(metadata, 'candidates_token_count', 0) or 0

    # Also check for cached tokens
    cached_tokens = getattr(metadata, 'cached_content_token_count', 0) or 0

    return (input_tokens + cached_tokens, output_tokens)


class ProductionManifest:
    """
    Tracks the completion status of pages and panels to allow for resumes.
    """
    def __init__(self, manifest_path):
        self.path = Path(manifest_path)
        self.data = self._load()

    def _load(self):
        if self.path.exists():
            with open(self.path, "r") as f:
                return json.load(f)
        return {"completed_pages": {}, "characters_designed": []}

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w") as f:
            json.dump(self.data, f, indent=2)

    def is_panel_complete(self, page_num, panel_id):
        page_key = str(page_num)
        if page_key in self.data["completed_pages"]:
            return str(panel_id) in self.data["completed_pages"][page_key]
        return False

    def mark_panel_complete(self, page_num, panel_id):
        page_key = str(page_num)
        if page_key not in self.data["completed_pages"]:
            self.data["completed_pages"][page_key] = []
        if str(panel_id) not in self.data["completed_pages"][page_key]:
            self.data["completed_pages"][page_key].append(str(panel_id))
        self.save()

    def is_character_designed(self, char_name):
        return char_name in self.data["characters_designed"]

    def mark_character_designed(self, char_name):
        if char_name not in self.data["characters_designed"]:
            self.data["characters_designed"].append(char_name)
        self.save()


def get_manifest_progress(manifest_path: str) -> dict:
    """
    Read ProductionManifest and return progress summary.

    Args:
        manifest_path: Path to the production_manifest.json file

    Returns:
        Dict with: pages_completed, panels_completed, characters_designed,
                   completed_page_numbers, error (if any)
    """
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        return {
            "pages_completed": 0,
            "panels_completed": 0,
            "characters_designed": 0,
            "completed_page_numbers": []
        }

    try:
        with open(manifest_path, "r") as f:
            data = json.load(f)

        completed_pages = data.get("completed_pages", {})
        panels_completed = sum(len(panels) for panels in completed_pages.values())
        pages_completed = len(completed_pages)
        characters_designed = len(data.get("characters_designed", []))

        return {
            "pages_completed": pages_completed,
            "panels_completed": panels_completed,
            "characters_designed": characters_designed,
            "completed_page_numbers": sorted([int(p) for p in completed_pages.keys()])
        }
    except Exception as e:
        return {
            "pages_completed": 0,
            "panels_completed": 0,
            "characters_designed": 0,
            "completed_page_numbers": [],
            "error": str(e)
        }


# ============================================================================
# BEAT DENSITY & PAGE ALLOCATION
# ============================================================================

def calculate_beat_density(beats: list, target_pages: int) -> dict:
    """
    Calculates page allocation based on beat intensity AND visual potential.

    Algorithm:
    1. Sum total intensity across all beats
    2. Calculate base pages per beat (target_pages / num_beats)
    3. Adjust allocation based on:
       - Intensity: High intensity beats (climax, crisis) get more pages
       - Visual Potential: High visual potential beats get more pages (they translate better)
       - Scene Type: Action/visual scenes get more pages than dialogue-heavy scenes
    4. Ensure minimum 1 page per beat
    5. Redistribute any rounding errors to highest combined score beats

    Args:
        beats: List of beat dicts with 'beat_id', 'beat_type', 'intensity', 'visual_potential'
        target_pages: Total pages to allocate

    Returns:
        Dict mapping beat_id to recommended page count
    """
    if not beats:
        return {}

    # Scene type modifiers for page allocation
    scene_modifiers = {
        "climax": 1.5,
        "crisis": 1.3,
        "midpoint": 1.2,
        "inciting": 1.1,
        "rising": 1.0,
        "falling": 0.9,
        "resolution": 0.9,
        "denouement": 0.8,
        "transition": 0.7
    }

    # Visual potential modifiers (beats that translate well to comics get more pages)
    visual_modifiers = {
        (0.0, 0.3): 0.7,   # Low visual potential - compress
        (0.3, 0.5): 0.85,  # Below average - slight compression
        (0.5, 0.7): 1.0,   # Average - neutral
        (0.7, 0.85): 1.15, # Good visual potential - expand slightly
        (0.85, 1.0): 1.3,  # Excellent visual potential - expand significantly
    }

    def get_visual_modifier(visual_potential: float) -> float:
        """Get the page modifier based on visual potential score."""
        for (low, high), modifier in visual_modifiers.items():
            if low <= visual_potential < high:
                return modifier
        return 1.0  # Default if visual_potential is exactly 1.0

    # Calculate total weighted intensity
    total_intensity = sum(b.get('intensity', 0.5) for b in beats)
    if total_intensity == 0:
        total_intensity = len(beats) * 0.5  # fallback

    pages_allocation = {}

    for beat in beats:
        beat_id = beat.get('beat_id', 0)
        intensity = beat.get('intensity', 0.5)
        beat_type = beat.get('beat_type', 'rising')
        visual_potential = beat.get('visual_potential', 0.5)

        # Weight by intensity relative to average
        weight = intensity / (total_intensity / len(beats)) if total_intensity > 0 else 1.0

        # Apply scene type modifier
        scene_modifier = scene_modifiers.get(beat_type, 1.0)

        # Apply visual potential modifier
        visual_modifier = get_visual_modifier(visual_potential)

        # Calculate pages for this beat (combine all modifiers)
        base_pages = target_pages / len(beats)
        allocated = max(1, round(base_pages * weight * scene_modifier * visual_modifier))
        pages_allocation[beat_id] = allocated

    # Redistribute to match target exactly
    current_total = sum(pages_allocation.values())
    difference = target_pages - current_total

    if difference != 0:
        # Sort beats by combined score (intensity + visual_potential) for redistribution
        def combined_score(b):
            return b.get('intensity', 0.5) + b.get('visual_potential', 0.5)

        sorted_beats = sorted(beats, key=combined_score, reverse=difference > 0)
        for i in range(abs(difference)):
            beat_id = sorted_beats[i % len(sorted_beats)].get('beat_id', i)
            if beat_id in pages_allocation:
                pages_allocation[beat_id] += 1 if difference > 0 else -1
                # Ensure minimum 1 page
                pages_allocation[beat_id] = max(1, pages_allocation[beat_id])

    return pages_allocation


# ============================================================================
# ERA-APPROPRIATE TERM REPLACEMENTS
# ============================================================================

ERA_TERM_REPLACEMENTS = {
    "1860s Victorian": {
        # Modern terms -> Period-appropriate
        "okay": "very well",
        "ok": "very well",
        "sure": "certainly",
        "yeah": "yes",
        "yep": "yes",
        "nope": "no",
        "phone": "telegram",
        "telephone": "telegram",
        "call me": "send word",
        "photo": "daguerreotype",
        "photograph": "daguerreotype",
        "camera": "photographic apparatus",
        "electricity": "galvanic force",
        "electric": "galvanic",
        "car": "carriage",
        "automobile": "carriage",
        "gun": "revolver",
        "pistol": "revolver",
        "flashlight": "lantern",
        "torch": "lantern",
        "scientist": "natural philosopher",
        "computer": "calculating engine",
        "machine": "apparatus",
        "plastic": "gutta-percha",
        "rubber": "india-rubber",
        "guy": "fellow",
        "guys": "gentlemen",
        "cool": "capital",
        "awesome": "extraordinary",
        "amazing": "remarkable",
        "incredible": "astonishing",
        "totally": "entirely",
        "definitely": "certainly",
        "basically": "essentially",
        "actually": "in truth",
        "literally": "truly",
        "stuff": "matters",
        "thing": "article",
        "things": "articles",
        "kid": "child",
        "kids": "children",
        "hang out": "spend time",
        "check out": "examine",
        "figure out": "determine",
        "find out": "discover",
        "show up": "arrive",
        "deal with": "attend to",
        "come up with": "devise",
        "get rid of": "dispose of",
    },
    "1920s Art Deco": {
        "okay": "swell",
        "cool": "the bee's knees",
        "awesome": "the cat's meow",
        "phone": "telephone",
        "car": "automobile",
        "guy": "fella",
        "guys": "fellas",
        "amazing": "swell",
        "incredible": "extraordinary",
    },
    "Medieval Fantasy": {
        "okay": "very well",
        "cool": "remarkable",
        "awesome": "magnificent",
        "amazing": "wondrous",
        "guy": "fellow",
        "guys": "men",
        "phone": "messenger",
        "car": "carriage",
        "gun": "crossbow",
        "electricity": "magic",
        "machine": "contraption",
    }
}


def normalize_era_term(term: str, era: str) -> str:
    """
    Replaces modern terms with period-appropriate alternatives.

    Args:
        term: Potentially anachronistic term
        era: Era identifier (e.g., "1860s Victorian")

    Returns:
        Period-appropriate replacement or original term
    """
    replacements = ERA_TERM_REPLACEMENTS.get(era, {})
    term_lower = term.lower()

    if term_lower in replacements:
        replacement = replacements[term_lower]
        # Preserve original capitalization pattern
        if term[0].isupper():
            return replacement.capitalize()
        return replacement

    return term


def fix_era_anachronisms(text: str, era: str) -> str:
    """
    Scans text for anachronistic terms and replaces them with era-appropriate alternatives.

    Args:
        text: Text to process (dialogue or narration)
        era: Era identifier for lookup

    Returns:
        Text with anachronisms replaced
    """
    import re

    replacements = ERA_TERM_REPLACEMENTS.get(era, {})
    if not replacements:
        return text

    result = text
    for modern_term, period_term in replacements.items():
        # Case-insensitive replacement with word boundaries
        pattern = re.compile(r'\b' + re.escape(modern_term) + r'\b', re.IGNORECASE)

        def replace_match(match):
            original = match.group(0)
            if original[0].isupper():
                return period_term.capitalize()
            return period_term

        result = pattern.sub(replace_match, result)

    return result


def calculate_page_count(word_count: int, test_mode: bool = False, user_override: int = None) -> dict:
    """
    Calculate optimal page count based on word count.

    Args:
        word_count: Number of words in source text
        test_mode: If True, return test mode defaults
        user_override: Optional user-specified page count

    Returns:
        Dict with: recommended, minimum, maximum, estimated_time_minutes,
        warning (str or None), density_category
    """
    from constants import (
        MIN_PAGES_PRODUCTION, MAX_PAGES_PRODUCTION, RECOMMENDED_MAX_PAGES,
        MIN_PAGES_TEST, DEFAULT_PAGES_TEST,
        DENSITY_SHORT_STORY, DENSITY_NOVELLA, DENSITY_NOVEL, DENSITY_EPIC,
        BRACKET_SHORT, BRACKET_NOVELLA, BRACKET_NOVEL
    )

    if test_mode:
        return {
            'recommended': DEFAULT_PAGES_TEST,
            'minimum': MIN_PAGES_TEST,
            'maximum': 10,
            'estimated_time_minutes': 5,
            'warning': None,
            'density_category': 'test'
        }

    # Determine density tier
    if word_count < BRACKET_SHORT:
        base_pages = word_count / DENSITY_SHORT_STORY
        category = "short_story"
    elif word_count < BRACKET_NOVELLA:
        base_pages = word_count / DENSITY_NOVELLA
        category = "novella"
    elif word_count < BRACKET_NOVEL:
        base_pages = word_count / DENSITY_NOVEL
        category = "novel"
    else:
        base_pages = word_count / DENSITY_EPIC
        category = "epic"

    # Apply constraints
    recommended = max(MIN_PAGES_PRODUCTION, min(RECOMMENDED_MAX_PAGES, int(base_pages)))
    minimum = max(MIN_PAGES_PRODUCTION, int(recommended * 0.6))
    maximum = min(MAX_PAGES_PRODUCTION, int(recommended * 1.5))

    # Time estimation (5 RPM for image gen, 4 panels/page avg)
    estimated_time_minutes = int((recommended * 4) / 5) + 25  # +25 for scripting/chars

    # Generate warnings
    warning = None
    if recommended >= 120:
        warning = f"Large project ({recommended} pages, ~{estimated_time_minutes//60}h {estimated_time_minutes%60}m). Consider fewer pages or splitting into volumes."

    if user_override:
        if user_override < minimum:
            warning = f"Warning: {user_override} pages is below recommended minimum ({minimum}). Story may feel rushed."
        elif user_override > maximum:
            warning = f"Warning: {user_override} pages exceeds recommended maximum ({maximum}). Long production time (~{int((user_override * 4) / 5) + 25} min)."
        recommended = user_override

    return {
        'recommended': recommended,
        'minimum': minimum,
        'maximum': maximum,
        'estimated_time_minutes': estimated_time_minutes,
        'warning': warning,
        'density_category': category
    }
