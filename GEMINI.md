# Project Constitution & Standards

## Tech Stack
- **Language:** Python 3.12+ (Strict type hinting required)
- **Primary AI:** Google Gemini API (Vertex AI or AI Studio)
- **Models:**
  - Text/Logic: `gemini-2.5-flash` (2M context window)
  - Image Primary: `gemini-3-pro-image-preview` (up to 4096px)
  - Image Fallback: `gemini-2.5-flash-preview-image`
  - Image Last Resort: `gemini-2.5-flash-image` (1024px)
- **Libraries:** `google-genai`, `Streamlit` (UI), `Pillow` (Image Layout), `ReportLab` (PDF generation)

## AI Behavior Rules
- **Response Format:** Prefer JSON for all data-heavy outputs between AI agents.
- **Code Style:** Use asynchronous programming (`asyncio`) for all API calls to enable parallel processing.
- **Character Consistency:** Always generate a "Character Reference Sheet" before starting panel generation. Pass reference images as PIL objects to the image generation API.
- **Safety:** Do not store API keys in code; use `.env` files.
- **Rate Limiting:** All API calls must use `retry_with_backoff()` decorator and respect `RateLimiter` semaphores.

## Creative Constraints
- Focus on "Cinematic Composition." Use standard comic layout math (Rule of Thirds).
- Text rendering should be handled by `Pillow` overlays, not the Image Model, for maximum legibility.
- Text bubbles require compositional negative space (uncluttered areas for readability).

## Backend Selection

| Feature | AI Studio | Vertex AI |
|---------|-----------|-----------|
| Setup | API key only | GCP project + ADC |
| Daily Limits | Yes (free tier) | No (pay-per-use) |
| Rate Limits | ~5 RPM | 10-30+ RPM |
| Production Ready | No | Yes |

See [VERTEX_AI_QUESTIONS.md](VERTEX_AI_QUESTIONS.md) for Vertex AI configuration details.
