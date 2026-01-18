"""
Core pipeline agents for Illustrative AI.

Usage:
    from agents import ScriptingAgent, PanelAgent, CompositorAgent, ReferenceAgent, LayoutAgent, ExportAgent

    # Backward compatibility alias
    from agents import IllustratorAgent  # Alias for PanelAgent
"""
from agents.scripting import ScriptingAgent
from agents.panel_agent import PanelAgent, IllustratorAgent  # IllustratorAgent is alias for backward compatibility
from agents.compositor_agent import CompositorAgent
from agents.reference_agent import ReferenceAgent
from agents.layout_agent import LayoutAgent
from agents.export_agent import ExportAgent

__all__ = [
    "ScriptingAgent",
    "PanelAgent",
    "IllustratorAgent",  # Backward compatibility alias for PanelAgent
    "CompositorAgent",
    "ReferenceAgent",
    "LayoutAgent",
    "ExportAgent",
]
