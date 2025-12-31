# 📚 Illustrative: AI Graphic Novel Engine

An automated pipeline to transform royalty-free literature into stylistically consistent graphic novels using the Gemini 2025 Multimodal Suite.

## 🎯 The Goal
To create a "Batch-First" application that ingests long-form public domain text and outputs a fully realized graphic novel (PDF/EPUB) based on user-defined style, tone, and length.

## 🚀 MVP (Phase 1)
- **Input:** A `.txt` file (Public Domain) + Style Prompt (e.g., "Manga") + Tone (e.g., "Dark Fantasy").
- **Output:** A 10-page PDF with structured panels, consistent character features, and dialogue overlays.
- **Engine:** Python-based CLI with Gemini 3.0 Flash (Scripting) and Gemini 3 Pro Image (Art).

## ⚙️ Environment Setup

### Prerequisites
- Python 3.12 or higher
- Google Gemini API key ([Get one here](https://aistudio.google.com/app/apikey))

### Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd illustrate-ai
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On macOS/Linux
   # or
   venv\Scripts\activate  # On Windows
   ```

3. **Install dependencies:**
   ```bash
   pip install streamlit google-generativeai pillow python-dotenv reportlab
   ```

4. **Set up environment variables:**

   Create a `.env` file in the project root:
   ```bash
   GEMINI_API_KEY=your_api_key_here
   ```

5. **Add font files (optional but recommended):**

   Place a TrueType font file (e.g., `PatrickHand-Regular.ttf`) in the `fonts/` directory for better text rendering in speech bubbles.

### Running the Application

**Test Mode (Streamlit UI - 1 page):**
```bash
streamlit run app.py
```
Then open your browser to the displayed URL (typically http://localhost:8501).

**Production Mode (100 pages):**
```bash
python production_run.py
```
Edit `production_run.py` to configure your input file, style, and tone before running.

### Project Structure

```
illustrate-ai/
├── app.py                      # Streamlit UI for interactive pipeline
├── production_run.py           # CLI for full production runs
├── scripting_agent.py          # Script generation and beat sheet creation
├── character_architect.py      # Character design and reference generation
├── illustrator_agent.py        # Panel image generation
├── compositor_agent.py         # Page layout and text overlay
├── continuity_validator.py     # Continuity checking across panels
├── exporter_agent.py           # PDF/EPUB export
├── utils.py                    # Retry logic, rate limiting, manifest tracking
├── assets/
│   ├── input/                  # Place source .txt files here
│   └── output/                 # Generated content (characters, pages, etc.)
└── fonts/                      # TrueType fonts for text rendering
```

### Testing

Run a quick smoke test:
```bash
# python smoke_test.py
ls -l venv/bin/python && pwd && ./venv/bin/python smoke_test.py
```