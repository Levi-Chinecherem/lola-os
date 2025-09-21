# Standard imports
import typing as tp

# Local
from .base import BaseTemplateAgent
from lola.core.state import State

"""
File: Defines the LegacyInterfaceAgent class for LOLA OS TMVP 1 Phase 2.

Purpose: Implements an agent for interfacing with legacy systems.
How: Uses LLM to generate API calls for legacy systems.
Why: Enables integration with old systems, per Developer Sovereignty tenet.
Full Path: lola-os/python/lola/agents/legacy_interface.py
Future Optimization: Migrate to Rust for secure legacy access (post-TMVP 1).
"""

class LegacyInterfaceAgent(BaseTemplateAgent):
    """LegacyInterfaceAgent: Interfaces with legacy systems. Does NOT persist state—use StateManager."""

    async def run(self, query: str) -> State:
        """
        Interface with legacy system based on query.

        Args:
            query: Input string for legacy interaction.

        Returns:
            State: State with legacy response.
        Does Not: Handle authentication—use key_manager.py.
        """
        self.state.update({"query": query})
        # Inline: Generate legacy API call with LLM
        api_prompt = f"Generate legacy API call for: {query}"
        api_call = await self._call_llm(api_prompt)
        # Inline: Execute API call (using api_client.py)
        tool_name = "api_client"
        params = {"url": "legacy-api.com", "method": "GET", "data": api_call}
        result = await self.execute_tool(tool_name, params)
        self.state.update({"output": result})
        return self.state

__all__ = ["LegacyInterfaceAgent"]