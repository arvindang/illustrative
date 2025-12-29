# 📋 Product Requirements Document (PRD)

## 1. User Inputs
- **Source:** Upload a book or provide a Project Gutenberg URL.
- **Style:** Selection (Noir, Watercolor, Cyberpunk, Classic Marvel, etc.).
- **Tone:** Selection (Somber, Energetic, Melancholic, Whimsical).
- **Target Length:** Number of pages or "Density" (panels per page).

## 2. Core Pipeline
1. **Ingestion Agent:** Cleans the text and identifies key narrative "beats."
2. **Screenplay Agent:** Converts beats into a JSON script (Panel description, Dialogue, Character list).
3. **Character Architect:** Generates visual descriptions and reference images for the "Main Cast."
4. **Illustration Agent:** Batches panel generation using the Gemini Batch API (cost-efficiency).
5. **Compositor Agent:** Stitches panels into pages, adds speech bubbles, and exports as PDF.

## 3. Success Metrics
- **Visual Continuity:** The protagonist must be recognizable across 90% of panels.
- **Narrative Flow:** The generated script must summarize the source material without losing core plot points.