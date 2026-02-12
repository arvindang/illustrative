"""
OpenAI client singleton.

Mirrors the pattern in utils.py:get_client() for the Gemini backend.
Provides a lazy-initialized AsyncOpenAI client configured from PipelineConfig.
"""
import openai

_client = None


def get_openai_client() -> openai.AsyncOpenAI:
    """
    Returns a cached AsyncOpenAI client.

    Reads the API key from config.openai_api_key (which sources from
    OPENAI_API_KEY env var). The client is created once and reused.
    """
    global _client

    if _client is None:
        from config import config
        if not config.openai_api_key:
            raise ValueError(
                "OpenAI API key not configured. Set OPENAI_API_KEY env var "
                "or config.openai_api_key."
            )
        _client = openai.AsyncOpenAI(api_key=config.openai_api_key)

    return _client


def reset_openai_client():
    """Reset the cached client. Useful when config changes at runtime."""
    global _client
    _client = None
