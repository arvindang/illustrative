# 📜 Project Constitution & Standards

## 🛠 Tech Stack
- **Language:** Python 3.12+ (Strict type hinting required).
- **Primary AI:** Google Gemini API (Vertex AI or AI Studio).
- **Models:** - Text/Logic: `gemini-1.5-pro` (2M context window).
    - Image: `gemini-3-pro-image` (utilizing Character Consistency tags).
- **Libraries:** `google-generativeai`, `FastAPI` (Backend), `Pillow` (Image Layout), `ReportLab` (PDF generation).

## 🤖 AI Behavior Rules
- **Response Format:** Prefer JSON for all data-heavy outputs between AI agents.
- **Code Style:** Use asynchronous programming (`asyncio`) for all API calls to enable batching.
- **Character Consistency:** Always generate a "Character Reference Sheet" before starting panel generation. Reference the `seed` or `reference_id` in every image prompt.
- **Safety:** Do not store API keys in code; use `.env` files.

## 🎨 Creative Constraints
- Focus on "Cinematic Composition." Use standard comic layout math (Rule of Thirds).
- Text rendering should be handled by `Pillow` overlays, not the Image Model, for maximum legibility.