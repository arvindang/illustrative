# 🏗 Architecture & Data Flow

## 🔄 Batch Processing Flow
The app uses a **Sequential-Parallel Hybrid** model:
1. **Sequential:** Ingest -> Script -> Character Reference.
2. **Parallel:** Panel Generation (All panels are sent to the Gemini Batch API simultaneously).
3. **Sequential:** Layout -> PDF Export.

## 📂 Directory Structure
- `/src/agents/`: Logic for Scripting, Artist, and Compositor.
- `/src/core/`: Gemini API wrappers and prompt templates.
- `/assets/input/`: Source books.
- `/assets/output/`: Generated panels and final PDFs.
- `/cache/`: Character reference images and JSON scripts to avoid redundant API calls.

## 📊 Data Schema (Panel JSON)
{
  "page_id": 1,
  "panel_id": 1,
  "visual_description": "Protagonist walking through a foggy London street, top-down view.",
  "dialogue": "I never thought I would return here.",
  "characters_present": ["Sherlock Holmes"],
  "layout_type": "full_width"
}