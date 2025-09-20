# Standard imports
import typing as tp
from abc import ABC, abstractmethod

# Local
from .base import BaseAgent
from lola.core.state import State
from lola.core.graph import StateGraph, Node
from lola.tools.base import BaseTool

"""
File: Defines the RouterAgent for LOLA OS TMVP 1 Phase 2.

Purpose: Routes queries to appropriate agents or tools.
How: Uses StateGraph to select paths via LLM decisions.
Why: Enables intelligent task delegation, per Choice by Design.
Full Path: lola-os/python/lola/agents/router.py
"""
class RouterAgent(BaseAgent):
    """RouterAgent: Directs queries to agents/tools. Does NOT persist state—use StateManager."""

    def __init__(self, tools: tp.List[BaseTool], model: str = "openai/gpt-4o"):
        """
        Initialize with tools and LLM model.

        Args:
            tools: List of BaseTool instances.
            model: LLM model string for litellm.
        """
        super().__init__(tools, model)
        self.graph = StateGraph(self.state)
        self.graph.add_node(Node(id="route", type="llm", function=self._route, description="Routing step"))

    async def run(self, query: str) -> State:
        """
        Route the query to appropriate agents/tools.

        Args:
            query: User input string.
        Returns:
            State: Updated state with routing results.
        Does Not: Persist state—caller must use StateManager.
        """
        self.state.update({"query": query})
        return await self.graph.execute()

    async def _route(self, state: State) -> dict:
        """Route the query using LLM."""
        prompt = f"Determine the best agent/tool for: {state.data.get('query')}"
        response = await self._call_llm(prompt)
        return {"route": response}