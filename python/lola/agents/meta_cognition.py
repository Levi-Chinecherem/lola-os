# Standard imports
import typing as tp
from abc import ABC, abstractmethod

# Local
from .base import BaseAgent
from lola.core.state import State
from lola.core.graph import StateGraph, Node
from lola.tools.base import BaseTool

"""
File: Defines the MetaCognitionAgent for LOLA OS TMVP 1 Phase 2.

Purpose: Implements an agent for self-reflection and optimization.
How: Uses StateGraph to analyze and improve its own performance via LLM.
Why: Enables adaptive agent behavior, per Choice by Design.
Full Path: lola-os/python/lola/agents/meta_cognition.py
"""
class MetaCognitionAgent(BaseAgent):
    """MetaCognitionAgent: Reflects and optimizes performance. Does NOT persist state—use StateManager."""

    def __init__(self, tools: tp.List[BaseTool], model: str = "openai/gpt-4o"):
        """
        Initialize with tools and LLM model.

        Args:
            tools: List of BaseTool instances.
            model: LLM model string for litellm.
        """
        super().__init__(tools, model)
        self.graph = StateGraph(self.state)
        self.graph.add_node(Node(id="reflect", type="llm", function=self._reflect, description="Reflection step"))

    async def run(self, query: str) -> State:
        """
        Run a meta-cognition cycle on the query.

        Args:
            query: User input string.
        Returns:
            State: Updated state with reflection results.
        Does Not: Persist state—caller must use StateManager.
        """
        self.state.update({"query": query})
        return await self.graph.execute()

    async def _reflect(self, state: State) -> dict:
        """Reflect on performance using LLM."""
        prompt = f"Reflect on task: {state.data.get('query')}"
        response = await self._call_llm(prompt)
        return {"reflection": response}