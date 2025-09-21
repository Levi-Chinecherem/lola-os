# Standard imports
import typing as tp

# Local
from .base import BaseTemplateAgent
from lola.core.state import State

"""
File: Defines the MetaCognitionAgent class for LOLA OS TMVP 1 Phase 2.

Purpose: Implements a meta-cognition agent for self-reflection and improvement.
How: Uses LLM to critique and refine outputs from other agents.
Why: Enhances agent performance through self-critique, per Choice by Design tenet.
Full Path: lola-os/python/lola/agents/meta_cognition.py
Future Optimization: Migrate to Rust for fast reflection loops (post-TMVP 1).
"""

class MetaCognitionAgent(BaseTemplateAgent):
    """MetaCognitionAgent: Implements self-reflection pattern. Does NOT persist state—use StateManager."""

    async def run(self, query: str) -> State:
        """
        Reflect on and refine a task output.

        Args:
            query: Input string with task output to reflect on.

        Returns:
            State: Refined state with improved output.
        Does Not: Handle group reflection—use orchestration/group_chat.py.
        """
        self.state.update({"query": query})
        # Inline: Critique with LLM
        critique_prompt = f"Critique and improve: {query}"
        critique = await self._call_llm(critique_prompt)
        self.state.update({"critique": critique})
        # Inline: Refine based on critique
        refine_prompt = f"Refine based on critique: {critique}"
        refined = await self._call_llm(refine_prompt)
        self.state.update({"output": refined})
        return self.state

__all__ = ["MetaCognitionAgent"]