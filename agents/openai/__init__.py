"""
OpenAI image generation backend for Illustrative AI.

Provides batch and sync image generation using OpenAI's API as an alternative
to the Gemini image pipeline. Supports character reference sheets and panel
generation with character consistency via reference images.

Usage:
    from agents.openai import OpenAIReferenceAgent, OpenAIPanelAgent
"""
from agents.openai.reference_agent import OpenAIReferenceAgent
from agents.openai.panel_agent import OpenAIPanelAgent

__all__ = [
    "OpenAIReferenceAgent",
    "OpenAIPanelAgent",
]
