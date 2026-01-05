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

def retry_with_backoff(max_retries=5, initial_delay=2, max_delay=60):
    """
    Decorator for retrying async functions with exponential backoff and jitter.
    Specifically targets 429 (ResourceExhausted) and 500/503 (Server Error).
    """
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            delay = initial_delay
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
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
    """Get or create the global TPM limiter."""
    global _global_tpm_limiter
    if _global_tpm_limiter is None:
        from config import config
        tpm_limit = getattr(config, 'tpm_limit', 1_000_000)
        tpm_safety_margin = getattr(config, 'tpm_safety_margin', 0.85)
        _global_tpm_limiter = TPMRateLimiter(
            tpm_limit=tpm_limit,
            safety_margin=tpm_safety_margin
        )
    return _global_tpm_limiter


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
