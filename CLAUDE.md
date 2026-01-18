# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Illustrative is an AI-powered graphic novel engine that transforms public domain literature into stylistically consistent graphic novels using the Gemini 2025 Multimodal Suite. The pipeline is fully automated and "batch-first," designed to handle long-form text adaptation.

## Core Architecture

The application follows a streamlined **3-agent pipeline**:

```
ScriptingAgent → IllustratorAgent → CompositorAgent
   (script +       (refs + panels)    (compose + export)
    assets)
```

### 1. ScriptingAgent (`scripting_agent.py`)
Uses Gemini Context Caching to load the full book, then runs a multi-pass process:
- **Pass 1 - Director**: Creates a page-by-page blueprint with summaries, mood, and focus text
- **Pass 1.5 - Asset Manifest**: Extracts characters and key objects with visual descriptions
- **Pass 2 - Scriptwriter**: Generates detailed panel scripts in parallel, referencing the cached book

Output: `_full_script.json` + `_assets.json`

### 2. IllustratorAgent (`illustrator_agent.py`)
Handles all image generation:
- **Reference Sheets**: Generates character and object reference images from the asset manifest
- **Panel Images**: Generates individual panel images using references for consistency
- Implements 3-tier fallback logic between image models

### 3. CompositorAgent (`compositor_agent.py`)
Handles final assembly and export:
- **Batch Layout**: Generates layouts for all pages in a single API call
- **Composition**: Assembles panels into pages with dynamic layouts, overlays speech bubbles and captions
- **Export**: Outputs to PDF/EPUB

## Key Technical Patterns

### API Management
- **Dual Backend Support**: Works with both Google AI Studio (API key) and Vertex AI (GCP project)
- Centralized client in `utils.py:get_client()` handles backend switching based on `GOOGLE_GENAI_USE_VERTEXAI` env var
- All Gemini API calls use `retry_with_backoff()` decorator for handling 429/500/503 errors with exponential backoff
- `RateLimiter` class (utils.py) implements semaphore-based RPM throttling
- Rate limits auto-adjust: AI Studio (5 RPM) vs Vertex AI (10-30 RPM based on quotas)
- All API operations are async using `asyncio` for parallel processing
- See `VERTEX_AI_QUESTIONS.md` for quota configuration and rate limit tuning

### State Management & Resume Capability
- `ProductionManifest` (utils.py) tracks completion status of pages, panels, and characters
- Manifest stored at `assets/output/production_manifest.json`
- Each agent checks manifest before processing to enable resuming interrupted runs
- Character deduplication registry prevents regenerating variants of the same character

### Model Configuration
- Text/Logic: `gemini-2.5-flash` (2M context window) for scripting and analysis
- Image generation (3-tier fallback):
  1. Primary: `gemini-3-pro-image-preview` (highest quality, up to 4096px)
  2. Fallback: `gemini-2.5-flash-preview-image`
  3. Last resort: `gemini-2.5-flash-image` (1024px, fastest)
- Character consistency uses reference images passed as PIL objects to image generation API
- All model names are Vertex AI compatible (no AI Studio aliases)

### Directory Structure
```
assets/
├── input/           # Source .txt files
└── output/
    ├── characters/  # Character metadata and reference images
    ├── pages/       # Individual panel images
    └── final_pages/ # Composed pages with text overlays
```

## Common Commands

### Running the Pipeline

**Test Mode (1 page/segment):**
```bash
streamlit run app.py
```
The Streamlit UI provides step-by-step execution with test mode enabled by default.

**Production Mode (100 pages):**
```bash
python production_run.py
```
Edit `production_run.py` to configure input file, style, and tone before running.

### Environment Setup

**First-time setup:**
```bash
python3 -m venv venv
source venv/bin/activate  # On macOS/Linux
pip install -r requirements.txt  # Note: requirements.txt may need to be created
```

**Required environment variables (.env file):**

Option 1 - AI Studio (simple, has daily limits):
```
GEMINI_API_KEY=your_api_key_here
```

Option 2 - Vertex AI (production, pay-per-use, no daily limits):
```
GOOGLE_CLOUD_PROJECT=your-gcp-project-id
GOOGLE_CLOUD_LOCATION=us-central1
GOOGLE_GENAI_USE_VERTEXAI=true
```
Then run: `gcloud auth application-default login`

### Individual Agent Testing

**Test scripting only:**
```bash
python smoke_test.py
```

**Test character/object reference generation:**
```python
from illustrator_agent import IllustratorAgent
illustrator = IllustratorAgent(assets_path="assets/output/my_book_assets.json")
await illustrator.generate_all_references(style="Botanical Illustration")
```

## Important Constraints

### Character Consistency
- Always generate character reference sheets BEFORE panel generation
- Character names are normalized to handle variants ("Professor Aronnax", "Pierre Aronnax", "Aronnax" → "Professor Aronnax")
- Reference images must be loaded as PIL objects and passed to image generation API
- Character metadata includes distinctive items that must persist across panels

### Visual Composition
- Text bubbles require compositional negative space (uncluttered areas)
- Bubble positions: "top-left", "top-right", "bottom-left", "bottom-right"
- Composition follows "rule of thirds" cinematic guidelines
- Text overlays handled by PIL, NOT by the image model for legibility

### Context Constraints
- Global context analysis determines era, technology level, and visual constants
- Context constraints passed to all agents to ensure historical/technical accuracy
- Example: "Setting: 1860s. Use period-accurate technology. Characters underwater MUST wear steampunk diving suits with copper helmets."

## Data Formats

### Script JSON Structure
```json
[
  {
    "page_number": 1,
    "panels": [
      {
        "panel_id": 1,
        "visual_description": "...",
        "characters": ["Captain Nemo", "Professor Aronnax"],
        "dialogue": "...",
        "bubble_position": "top-left"
      }
    ]
  }
]
```

### Character Metadata Structure
```json
{
  "name": "Captain Nemo",
  "description": "...",
  "distinctive_items": ["ornate harpoon", "naval uniform"],
  "reference_images": ["assets/output/characters/captain_nemo/ref_1.png"]
}
```

## Development Notes

- Test mode uses 10k character slice from the middle of source text to save tokens
- Production mode processes full text (can be 100+ pages)
- Font files should be placed in `fonts/` directory (e.g., `PatrickHand-Regular.ttf`)
- Manifest-based resume means you can interrupt and restart production runs
- Character bank is loaded into memory as PIL images before panel generation
