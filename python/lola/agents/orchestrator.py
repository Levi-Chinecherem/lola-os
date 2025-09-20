# Standard imports
import typing as tp
from abc import ABC, abstractmethod

# Local
from .base import BaseAgent
from lola.core.state import State
from lola.core.graph import StateGraph, Node
from lola.tools.base import BaseTool

"""
File: Defines the OrchestratorAgent for LOLA OS TMVP 1 Phase 2.

Purpose: Implements a master agent for managing multiple agents.
How: Uses StateGraph to orchestrate agent workflows via LLM.
Why: Enables complex multi-agent coordination, per Choice by Design.
Full Path: lola-os/python/lola/agents/orchestrator.py
"""
class OrchestratorAgent(BaseAgent):
    """OrchestratorAgent: Manages multiple agents. Does NOT persist state—use StateManager."""

    def __init__(self, tools: tp.List[BaseTool], model: str = "openai/gpt-4o"):
        """
        Initialize with tools and LLM model.

        Args:
            tools: List of BaseTool instances.
            model: LLM model string for litellm.
        """
        super().__init__(tools, model)
        self.graph = StateGraph(self.state)
        self.graph.add_node(Node(id="orchestrate", type="llm", function=self._orchestrate, description="Orchestration step"))

    async def run(self, query: str) -> State:
        """
        Run an orchestration cycle on the query.

        Args:
            query: User input string.
        Returns:
            State: Updated state with orchestration results.
        Does Not: Persist state—caller must use StateManager.
        """
        self.state.update({"query": query})
        return await self.graph.execute()

    async def _orchestrate(self, state: State) -> dict:
        """Orchestrate agents using LLM."""
        prompt = f"Orchestrate tasks for: {state.data.get('query')}"
        response = await self._call_llm(prompt)
        return {"orchestration": response}