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

### 1. ScriptingAgent (`agents/scripting_agent.py`)
Uses Gemini Context Caching to load the full book (2M context window), then runs a **7-pass enrichment pipeline**:

```
PASS 0: Global Context (user-provided era constraints)
PASS 1: Beat Analysis → _beats.json
        - Narrative beats with intensity + visual_potential scores
        - Page-turn hooks and cliffhanger identification
        - Micro-beats for panel-level pacing
PASS 1.5: Adaptation Filter → _adaptation.json
        - Essential/condensable/cuttable scene classification
        - Reader-beloved moments (must-keep fan favorites)
        - Pacing recommendations (slow_down/speed_up beats)
PASS 2: Director Pass → _blueprint.json
        - Page-by-page blueprint with spread awareness (two-page spreads)
        - Cliffhanger marking (odd pages for page-turn suspense)
        - suggested_panel_count and recommended_splash flags
PASS 3: Character Deep Dive → _character_arcs.json
        - Arc typing (transformation, flat, redemption, etc.)
        - Voice profiles with dialect markers, catchphrases, dialogue samples
        - Scene-specific gear and emotional states
PASS 4: Asset Manifest → _assets.json
        - Characters with color_signature for visual identity
        - Recurring locations with lighting, mood, color_palette
        - Global color_script for act-based color direction
        - Interaction rules per scene type
PASS 5: Scriptwriter (Parallel) → _full_script.json
        - Cinematic shot_type (establishing, close-up, over-shoulder, etc.)
        - panel_size (large/medium/small) for pacing control
        - Adjacent page context for scene continuity
        - Voice profile integration for dialogue consistency
        - Retry logic for failed pages (up to 2 retries)
PASS 6: Validation + Auto-Fix → _validation.json
        - Pre-validation via PromptPreValidator (saves tokens)
        - Era anachronism detection and replacement
        - Dialogue/caption length enforcement
```

**Key Handoffs Between Passes:**
- Beat analysis → Director: visual_potential scores inform page allocation
- Adaptation filter → Director: essential/condensable guidance
- Character arcs → Scriptwriter: voice profiles ensure dialogue consistency
- Blueprint → Scriptwriter: adjacent page context (prev/next) for continuity
- Chunk handoff in Director: last 8 pages with full detail (characters, mood, scene type)

Output: `_full_script.json` + `_assets.json` + `_beats.json` + `_character_arcs.json` + `_adaptation.json` + `_validation.json`

### 2. IllustratorAgent (`agents/illustrator_agent.py`)
Handles all image generation:
- **Reference Sheets**: Generates character and object reference images from the asset manifest
- **Panel Images**: Generates individual panel images using references for consistency
- Implements 3-tier fallback logic between image models

### 3. CompositorAgent (`agents/compositor_agent.py`)
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
- See `docs/VERTEX_AI_QUESTIONS.md` for quota configuration and rate limit tuning

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

### Project Structure
```
illustrate-ai/
├── agents/              # Core pipeline agents
│   ├── scripting_agent.py
│   ├── illustrator_agent.py
│   └── compositor_agent.py
├── models/              # SQLAlchemy models (User, Novel)
├── migrations/          # Alembic database migrations
├── storage/             # S3/bucket storage utilities
├── tests/               # pytest test suite
├── scripts/             # Utility/debug scripts
├── docs/                # Documentation (PRD, GEMINI.md, etc.)
├── deploy/              # Alembic config for migrations
├── assets/
│   ├── input/           # Source .txt files
│   └── output/          # Generated content
│       ├── characters/  # Character reference images
│       ├── pages/       # Individual panel images
│       └── final_pages/ # Composed pages with text overlays
├── app.py               # Streamlit UI entry point
├── production_run.py    # CLI production entry point
├── config.py            # Configuration
├── constants.py         # Constants (styles, tones)
├── utils.py             # Shared utilities
└── validators.py        # Validation logic
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

### Access Control (Admin Allowlist)

The app uses a hosted Vertex AI backend (no user API keys). To control costs, access can be restricted to specific email addresses using `ADMIN_EMAILS`.

**Environment variable:**
```
ADMIN_EMAILS=arvin@arvindang.com
```

**Behavior:**
| ADMIN_EMAILS value | Behavior |
|-------------------|----------|
| Not set / empty | Open access (anyone can register/login) |
| Single email | Only that email can login; registration blocked |
| Comma-separated | Multiple admins allowed |

**Railway CLI commands:**
```bash
# View current setting
railway variables | grep ADMIN_EMAILS

# Restrict to single admin
railway variables --set "ADMIN_EMAILS=arvin@arvindang.com"

# Add multiple admins
railway variables --set "ADMIN_EMAILS=arvin@arvindang.com,other@example.com"

# Open access to everyone (remove restriction)
railway variables --set "ADMIN_EMAILS="
```

**Implementation:** `config.py` (`PipelineConfig.admin_emails`, `is_admin_email()`, `get_admin_emails()`)

### Testing

**Run all tests:**
```bash
pytest tests/ -v
```

**Test scripting only (JSON generation, no images):**
```bash
pytest tests/test_scripting_only.py -v
```

**Test full pipeline:**
```bash
pytest tests/test_full_pipeline.py -v
```

**Custom test options:**
```bash
pytest tests/test_scripting_only.py --input-file assets/input/alice.txt --style "Lush Watercolor" --pages 3 -v
```

**Test character/object reference generation:**
```python
from agents import IllustratorAgent
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
        "caption": "...",
        "bubble_position": "top-left",
        "shot_type": "medium",  // establishing, wide, medium, close-up, extreme-close-up, over-shoulder, two-shot, birds-eye, worms-eye
        "panel_size": "medium", // large (50%+), medium (25-50%), small (15-25%)
        "key_objects": ["The Nautilus", "Harpoon"],
        "advice": {
          "scene_type": "dialogue",
          "required_gear": {"Captain Nemo": ["naval uniform"]},
          "era_constraints": ["Victorian clothing"],
          "continuity": {"from_previous": "...", "to_next": "..."},
          "composition": {"negative_space": "top-left"}
        }
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

- Test mode uses a reduced page count (default 3 pages) but loads the full source text for context caching. The full text is needed for accurate beat analysis and adaptation.
- Production mode processes full text (can be 100+ pages)
- Font files should be placed in `fonts/` directory (e.g., `PatrickHand-Regular.ttf`)
- Manifest-based resume means you can interrupt and restart production runs
- Character bank is loaded into memory as PIL images before panel generation
