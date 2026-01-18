"""
Scripting passes for the graphic novel pipeline.

Each pass is an async function that takes a ScriptingContext and returns its output.
"""
from agents.scripting.passes.beat_analysis import analyze_narrative_beats
from agents.scripting.passes.adaptation_filter import generate_adaptation_filter
from agents.scripting.passes.pacing_blueprint import generate_pacing_blueprint
from agents.scripting.passes.character_deep_dive import generate_character_deep_dive
from agents.scripting.passes.asset_manifest import generate_asset_manifest
from agents.scripting.passes.scriptwriter import write_page_script
from agents.scripting.passes.dialogue_polish import polish_dialogue

__all__ = [
    "analyze_narrative_beats",
    "generate_adaptation_filter",
    "generate_pacing_blueprint",
    "generate_character_deep_dive",
    "generate_asset_manifest",
    "write_page_script",
    "polish_dialogue",
]
