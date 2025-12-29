# 🗺 Development Roadmap

## Phase 1: Ingestion & Scripting (Week 1)
- [ ] Setup Python environment and API authentication.
- [ ] Build `TextCleaner` to strip Gutenberg headers/footers.
- [ ] Create `ScriptingAgent` to convert Chapter 1 into JSON panels.

## Phase 2: Visual Foundation (Week 2)
- [ ] Implement `CharacterArchitect` to generate consistent protagonist IDs.
- [ ] Test Gemini 3 Image API with style/tone modifiers.
- [ ] Build a "Previewer" script (Outputs basic HTML to see images + text).

## Phase 3: Batch & Layout (Week 3)
- [ ] Integrate Gemini Batch API for cost-effective generation.
- [ ] Create `Compositor` using Pillow to draw speech bubbles over images.
- [ ] Implement PDF export via ReportLab.

## Phase 4: Polish (Week 4)
- [ ] Build a Streamlit UI for non-technical testing.
- [ ] Add "Tone Validation" (Does the image match the text's mood?).