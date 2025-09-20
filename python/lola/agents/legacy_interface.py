# Standard imports
import typing as tp
from abc import ABC, abstractmethod

# Local
from .base import BaseAgent
from lola.core.state import State
from lola.core.graph import StateGraph, Node
from lola.tools.base import BaseTool

"""
File: Defines the LegacyInterfaceAgent for LOLA OS TMVP 1 Phase 2.

Purpose: Implements an agent for interfacing with legacy systems.
How: Uses StateGraph to translate queries for legacy APIs via LLM.
Why: Enables integration with existing systems, per Developer Sovereignty.
Full Path: lola-os/python/lola/agents/legacy_interface.py
"""
class LegacyInterfaceAgent(BaseAgent):
    """LegacyInterfaceAgent: Interfaces with legacy systems. Does NOT persist state—use StateManager."""

    def __init__(self, tools: tp.List[BaseTool], model: str = "openai/gpt-4o"):
        """
        Initialize with tools and LLM model.

        Args:
            tools: List of BaseTool instances.
            model: LLM model string for litellm.
        """
        super().__init__(tools, model)
        self.graph = StateGraph(self.state)
        self.graph.add_node(Node(id="interface", type="llm", function=self._interface, description="Legacy interface step"))

    async def run(self, query: str) -> State:
        """
        Run a legacy interface cycle on the query.

        Args:
            query: User input string.
        Returns:
            State: Updated state with interface results.
        Does Not: Persist state—caller must use StateManager.
        """
        self.state.update({"query": query})
        return await self.graph.execute()

    async def _interface(self, state: State) -> dict:
        """Interface with legacy systems using LLM."""
        prompt = f"Translate query for legacy system: {state.data.get('query')}"
        response = await self._call_llm(prompt)
        return {"interface": response}