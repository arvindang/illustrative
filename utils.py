import asyncio
import random
import functools
import json
from pathlib import Path
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
