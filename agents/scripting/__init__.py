"""
Scripting package for the graphic novel pipeline.

Exports the main ScriptingAgent class and ScriptingContext dataclass.
"""
from agents.scripting.agent import ScriptingAgent
from agents.scripting.utils import ScriptingContext

__all__ = ["ScriptingAgent", "ScriptingContext"]
