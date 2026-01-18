# Illustrative: AI Graphic Novel Engine

An automated pipeline to transform royalty-free literature into stylistically consistent graphic novels using the Gemini 2025 Multimodal Suite.

## The Goal

A "Batch-First" application that ingests long-form public domain text and outputs a fully realized graphic novel (PDF/EPUB) based on user-defined style, tone, and length.

## Quick Start

### Prerequisites
- Python 3.12 or higher
- Either a Google Gemini API key OR a Google Cloud project with Vertex AI enabled

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
   pip install -r requirements.txt
   ```

4. **Configure API access** (choose one):

   **Option A - AI Studio (simple, has daily limits):**
   ```bash
   # Create .env file
   echo "GEMINI_API_KEY=your_api_key_here" > .env
   ```
   Get an API key at [aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)

   **Option B - Vertex AI (production, pay-per-use, no daily limits):**
   ```bash
   # Create .env file
   cat > .env << EOF
   GOOGLE_CLOUD_PROJECT=your-gcp-project-id
   GOOGLE_CLOUD_LOCATION=us-central1
   GOOGLE_GENAI_USE_VERTEXAI=true
   EOF

   # Authenticate
   gcloud auth application-default login
   ```
   See [VERTEX_AI_QUESTIONS.md](VERTEX_AI_QUESTIONS.md) for quota configuration.

5. **Add font files (optional but recommended):**
   Place a TrueType font file (e.g., `PatrickHand-Regular.ttf`) in the `fonts/` directory for better text rendering in speech bubbles.

### Running the Application

**Test Mode (Streamlit UI):**
```bash
streamlit run app.py
```
Then open your browser to the displayed URL (typically http://localhost:8501).

**Production Mode (full book processing):**
```bash
python production_run.py
```
Edit `production_run.py` to configure your input file, style, and tone before running.

## Architecture

The pipeline uses a 3-agent architecture to transform text into graphic novels:

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         ILLUSTRATIVE AI PIPELINE                                │
└─────────────────────────────────────────────────────────────────────────────────┘

    ┌──────────────┐
    │  INPUT FILE  │   Public domain literature (.txt)
    │   (Upload)   │   e.g., Alice in Wonderland, 20,000 Leagues Under the Sea
    └──────┬───────┘
           │
           ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│  SCRIPTING AGENT                                                                 │
│  ════════════════                                                                │
│  Uses Gemini 2M context window to analyze full book                              │
│                                                                                  │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐          │
│  │  PASS 0-1   │──▶│  PASS 1.5   │──▶│  PASS 2-3   │──▶│  PASS 4-6   │          │
│  │  Beat       │   │  Adaptation │   │  Director + │   │  Assets +   │          │
│  │  Analysis   │   │  Filter     │   │  Characters │   │  Script     │          │
│  └─────────────┘   └─────────────┘   └─────────────┘   └─────────────┘          │
│                                                                                  │
│  Output: _full_script.json, _assets.json, _beats.json, _character_arcs.json     │
└──────────────────────────────────────────────────────────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│  ILLUSTRATOR AGENT                                                               │
│  ═════════════════                                                               │
│                                                                                  │
│  ┌─────────────────────────┐        ┌─────────────────────────────────┐         │
│  │   REFERENCE SHEETS      │        │      PANEL GENERATION           │         │
│  │   ─────────────────     │        │      ────────────────           │         │
│  │   • Character refs      │───────▶│   • Uses refs for consistency   │         │
│  │   • Object refs         │        │   • 3-tier model fallback       │         │
│  │   • Location refs       │        │   • Parallel async generation   │         │
│  └─────────────────────────┘        └─────────────────────────────────┘         │
│                                                                                  │
│  Output: assets/output/characters/*.png, assets/output/pages/*.png              │
└──────────────────────────────────────────────────────────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│  COMPOSITOR AGENT                                                                │
│  ════════════════                                                                │
│                                                                                  │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐      │
│  │    LAYOUT    │──▶│   COMPOSE    │──▶│    TEXT      │──▶│    EXPORT    │      │
│  │   ────────   │   │   ───────    │   │   ────       │   │   ──────     │      │
│  │  Panel grid  │   │  Assemble    │   │  Speech      │   │  • PDF       │      │
│  │  positions   │   │  panels on   │   │  bubbles +   │   │  • EPUB      │      │
│  │  and sizes   │   │  page canvas │   │  captions    │   │              │      │
│  └──────────────┘   └──────────────┘   └──────────────┘   └──────────────┘      │
│                                                                                  │
│  Output: assets/output/final_pages/*.png                                        │
└──────────────────────────────────────────────────────────────────────────────────┘
           │
           ▼
    ┌──────────────┐
    │    OUTPUT    │   Complete graphic novel ready for reading
    │  (PDF/EPUB)  │   with consistent characters and professional layouts
    └──────────────┘
```

See [CLAUDE.md](CLAUDE.md) for detailed architecture documentation.

## Project Structure

```
illustrate-ai/
├── app.py                  # Streamlit UI for interactive pipeline
├── production_run.py       # CLI for full production runs
├── scripting_agent.py      # Script generation and beat sheet creation
├── illustrator_agent.py    # Character refs + panel image generation
├── compositor_agent.py     # Page layout, text overlay, and export
├── utils.py                # Retry logic, rate limiting, manifest tracking
├── config.py               # Centralized configuration
├── assets/
│   ├── input/              # Place source .txt files here
│   └── output/             # Generated content (characters, pages, etc.)
└── fonts/                  # TrueType fonts for text rendering
```

## Testing

**Quick smoke test:**
```bash
python smoke_test.py
```

**Validate configuration:**
```bash
python config.py
```

## Troubleshooting

See [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for common errors and solutions.

## Documentation

- [CLAUDE.md](CLAUDE.md) - Architecture and development guide (for Claude Code)
- [GEMINI.md](GEMINI.md) - AI behavior rules and creative constraints
- [VERTEX_AI_QUESTIONS.md](VERTEX_AI_QUESTIONS.md) - Vertex AI quota and configuration guide
- [docs/PRD.md](docs/PRD.md) - Product requirements document
