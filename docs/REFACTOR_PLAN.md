# Image Pipeline Refactor Plan

## Goal
Split monolithic agent files for better agentic flows: parallelization, resume granularity, testing, and debugging.

## Current State

| File | LOC | Classes | Responsibilities |
|------|-----|---------|-----------------|
| `illustrator_agent.py` | 1,101 | 1 | Reference sheets + Panel generation + Caching |
| `compositor_agent.py` | 987 | 1 | Layout + Text overlay + Export |
| `validators.py` | 1,388 | 13 | Pre-validation + Post-validation + Composition + Consistency |
| `utils.py` | 873 | 5 | Rate limiting + Token tracking + Manifest + Retry |

## Target Structure

```
agents/
├── reference_agent.py      # Character & object reference generation
│   └── ReferenceAgent      # ~400 LOC, parallelizable per character
│
├── panel_agent.py          # Panel image generation only
│   └── PanelAgent          # ~500 LOC, sequential with continuity
│
├── layout_agent.py         # Page layout generation (LLM-powered)
│   └── LayoutAgent         # ~300 LOC, batch-capable
│
├── compositor_agent.py     # Text overlay & assembly (deterministic)
│   └── CompositorAgent     # ~400 LOC, pure PIL operations
│
├── export_agent.py         # PDF/EPUB export
│   └── ExportAgent         # ~200 LOC, output format specific
│
└── __init__.py

validators/
├── __init__.py
├── pre_validators.py       # PromptPreValidator, EraConstraintChecker
├── post_validators.py      # PanelValidator, OutputQualityChecker
├── composition.py          # ImageCompositionAnalyzer
└── consistency.py          # ConsistencyAuditor
```

## Migration Tasks

### Phase 1: Split validators.py (High Priority)
- [ ] Create `validators/` directory
- [ ] Extract `PromptPreValidator`, `EraConstraintChecker` → `pre_validators.py`
- [ ] Extract `PanelValidator`, `OutputQualityChecker` → `post_validators.py`
- [ ] Extract `ImageCompositionAnalyzer` → `composition.py`
- [ ] Extract `ConsistencyAuditor` → `consistency.py`
- [ ] Create `validators/__init__.py` with re-exports
- [ ] Update imports in `illustrator_agent.py` and `compositor_agent.py`
- [ ] Delete old `validators.py`

### Phase 2: Extract ReferenceAgent (High Priority)
- [ ] Create `agents/reference_agent.py`
- [ ] Move reference generation methods from IllustratorAgent:
  - `generate_character_reference()`
  - `generate_object_reference()`
  - `generate_all_references()`
  - `_select_best_reference()`
  - Reference loading/caching methods
- [ ] Update IllustratorAgent to use ReferenceAgent or accept pre-loaded refs
- [ ] Ensure parallel character reference generation works

### Phase 3: Split CompositorAgent (Medium Priority)
- [ ] Create `agents/layout_agent.py` - LLM-powered layout generation
- [ ] Create `agents/export_agent.py` - PDF/EPUB export
- [ ] Slim down `compositor_agent.py` to text overlay + assembly only

### Phase 4: Rename IllustratorAgent → PanelAgent (Medium Priority)
- [ ] Rename file to `panel_agent.py`
- [ ] Rename class to `PanelAgent`
- [ ] Update all imports

## Key Interface Pattern

Each agent follows the same contract for orchestration:

```python
class ReferenceAgent:
    async def run(self, manifest: ProductionManifest) -> None:
        """Generate all pending references, update manifest"""

class PanelAgent:
    async def run(self, manifest: ProductionManifest) -> None:
        """Generate all pending panels, update manifest"""

class LayoutAgent:
    async def run(self, manifest: ProductionManifest) -> None:
        """Generate layouts for all pending pages"""

class CompositorAgent:
    async def run(self, manifest: ProductionManifest) -> None:
        """Compose all pending pages with text overlays"""

class ExportAgent:
    async def run(self, manifest: ProductionManifest, format: str) -> Path:
        """Export to PDF/EPUB"""
```

## Orchestration Example

```python
# Parallel reference generation
await reference_agent.run(manifest)  # All characters in parallel internally

# Sequential panel generation (for visual continuity)
await panel_agent.run(manifest)

# Batch layout + compose
await layout_agent.run(manifest)
await compositor_agent.run(manifest)
await export_agent.run(manifest)
```

## Benefits

| Benefit | Current | Proposed |
|---------|---------|----------|
| Parallelization | Reference + panel coupled | Reference generation fully parallelizable |
| Resume granularity | Page-level | Task-level (refs, panels, layout, compose) |
| Testing | Integration tests required | Unit tests per agent |
| Debugging | Search 1,100 LOC file | Search ~400 LOC file |
| Retry isolation | Failure retries whole phase | Retry only failed sub-task |
| Model selection | One fallback chain | Per-agent model tuning |
