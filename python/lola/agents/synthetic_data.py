# Standard imports
import typing as tp
from abc import ABC, abstractmethod

# Local
from .base import BaseAgent
from lola.core.state import State
from lola.core.graph import StateGraph, Node
from lola.tools.base import BaseTool

"""
File: Defines the SyntheticDataAgent for LOLA OS TMVP 1 Phase 2.

Purpose: Implements an agent for generating synthetic data.
How: Uses StateGraph to create synthetic datasets via LLM.
Why: Enables data generation for training/testing, per Developer Sovereignty.
Full Path: lola-os/python/lola/agents/synthetic_data.py
"""
class SyntheticDataAgent(BaseAgent):
    """SyntheticDataAgent: Generates synthetic data. Does NOT persist state—use StateManager."""

    def __init__(self, tools: tp.List[BaseTool], model: str = "openai/gpt-4o"):
        """
        Initialize with tools and LLM model.

        Args:
            tools: List of BaseTool instances.
            model: LLM model string for litellm.
        """
        super().__init__(tools, model)
        self.graph = StateGraph(self.state)
        self.graph.add_node(Node(id="generate", type="llm", function=self._generate, description="Data generation step"))

    async def run(self, query: str) -> State:
        """
        Run a synthetic data generation cycle on the query.

        Args:
            query: User input string.
        Returns:
            State: Updated state with generated data.
        Does Not: Persist state—caller must use StateManager.
        """
        self.state.update({"query": query})
        return await self.graph.execute()

    async def _generate(self, state: State) -> dict:
        """Generate synthetic data using LLM."""
        prompt = f"Generate synthetic data for: {state.data.get('query')}"
        response = await self._call_llm(prompt)
        return {"data": response}