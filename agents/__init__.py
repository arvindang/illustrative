"""
Core pipeline agents for Illustrative AI.

Usage:
    from agents import ScriptingAgent, PanelAgent, CompositorAgent, ReferenceAgent, LayoutAgent, ExportAgent

    # Backward compatibility alias
    from agents import IllustratorAgent  # Alias for PanelAgent

    # Factory for image backend selection (Gemini vs OpenAI)
    from agents import get_image_agents
"""
from pathlib import Path

from agents.scripting import ScriptingAgent
from agents.panel_agent import PanelAgent, IllustratorAgent  # IllustratorAgent is alias for backward compatibility
from agents.compositor_agent import CompositorAgent
from agents.reference_agent import ReferenceAgent
from agents.layout_agent import LayoutAgent
from agents.export_agent import ExportAgent


def get_image_agents(script_path: str, style_prompt: str, base_output_dir: Path = None):
    """
    Factory function that returns the appropriate (ReferenceAgent, PanelAgent)
    pair based on config.image_backend.

    Args:
        script_path: Path to the full script JSON.
        style_prompt: Global style directive for image generation.
        base_output_dir: Optional base output directory override.

    Returns:
        Tuple of (reference_agent, panel_agent) matching the configured backend.
    """
    from config import config

    script_p = Path(script_path)
    stem = script_p.stem.replace("_full_script", "").replace("_test_page", "")
    assets_path = script_p.parent / f"{stem}_assets.json"
    assets_str = str(assets_path) if assets_path.exists() else None

    if config.image_backend == "openai":
        from agents.openai import OpenAIReferenceAgent, OpenAIPanelAgent

        ref_agent = OpenAIReferenceAgent(
            assets_path=assets_str,
            base_output_dir=base_output_dir,
            style_prompt=style_prompt,
        )
        panel_agent = OpenAIPanelAgent(
            script_path=script_path,
            style_prompt=style_prompt,
            base_output_dir=base_output_dir,
            reference_agent=ref_agent,
        )
        return ref_agent, panel_agent
    else:
        ref_agent = ReferenceAgent(
            assets_path=assets_str,
            base_output_dir=base_output_dir,
            style_prompt=style_prompt,
        )
        panel_agent = PanelAgent(
            script_path=script_path,
            style_prompt=style_prompt,
            base_output_dir=base_output_dir,
            reference_agent=ref_agent,
        )
        return ref_agent, panel_agent


__all__ = [
    "ScriptingAgent",
    "PanelAgent",
    "IllustratorAgent",  # Backward compatibility alias for PanelAgent
    "CompositorAgent",
    "ReferenceAgent",
    "LayoutAgent",
    "ExportAgent",
    "get_image_agents",
]
