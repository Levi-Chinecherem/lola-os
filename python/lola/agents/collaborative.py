# Standard imports
import typing as tp
from abc import ABC, abstractmethod

# Local
from .base import BaseAgent
from lola.core.state import State
from lola.core.graph import StateGraph, Node
from lola.tools.base import BaseTool

"""
File: Defines the CollaborativeAgent for LOLA OS TMVP 1 Phase 2.

Purpose: Implements a multi-agent collaborative workflow.
How: Uses StateGraph to coordinate multiple agents via LLM.
Why: Enables teamwork among agents, per Choice by Design.
Full Path: lola-os/python/lola/agents/collaborative.py
"""
class CollaborativeAgent(BaseAgent):
    """CollaborativeAgent: Coordinates multiple agents. Does NOT persist state—use StateManager."""

    def __init__(self, tools: tp.List[BaseTool], model: str = "openai/gpt-4o"):
        """
        Initialize with tools and LLM model.

        Args:
            tools: List of BaseTool instances.
            model: LLM model string for litellm.
        """
        super().__init__(tools, model)
        self.graph = StateGraph(self.state)
        self.graph.add_node(Node(id="coordinate", type="llm", function=self._coordinate, description="Coordination step"))

    async def run(self, query: str) -> State:
        """
        Run a collaborative cycle on the query.

        Args:
            query: User input string.
        Returns:
            State: Updated state with coordination results.
        Does Not: Persist state—caller must use StateManager.
        """
        self.state.update({"query": query})
        return await self.graph.execute()

    async def _coordinate(self, state: State) -> dict:
        """Coordinate among agents using LLM."""
        prompt = f"Coordinate tasks for: {state.data.get('query')}"
        response = await self._call_llm(prompt)
        return {"coordination": response}