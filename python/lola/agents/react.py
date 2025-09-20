# Standard imports
import typing as tp
from abc import ABC, abstractmethod

# Local
from .base import BaseAgent
from lola.core.state import State
from lola.core.graph import StateGraph, Node
from lola.tools.base import BaseTool

"""
File: Defines the ReActAgent for LOLA OS TMVP 1 Phase 2.

Purpose: Implements a ReAct (Reasoning-Acting) agent for complex problem-solving.
How: Uses StateGraph to alternate between LLM reasoning and tool actions.
Why: Enables structured reasoning loops, per Explicit over Implicit.
Full Path: lola-os/python/lola/agents/react.py
"""
class ReActAgent(BaseAgent):
    """ReActAgent: Orchestrates reasoning-acting cycles. Does NOT persist state—use StateManager."""

    def __init__(self, tools: tp.List[BaseTool], model: str = "openai/gpt-4o"):
        """
        Initialize with tools and LLM model.

        Args:
            tools: List of BaseTool instances.
            model: LLM model string for litellm.
        """
        super().__init__(tools, model)
        self.graph = StateGraph(self.state)
        self.graph.add_node(Node(id="reason", type="llm", function=self._reason, description="Reasoning step"))
        self.graph.add_node(Node(id="act", type="tool", function=self._act, description="Tool execution"))

    async def run(self, query: str) -> State:
        """
        Run a ReAct cycle on the query.

        Args:
            query: User input string.
        Returns:
            State: Updated state with thoughts/actions.
        Does Not: Persist state—caller must use StateManager.
        """
        self.state.update({"query": query})
        return await self.graph.execute()

    async def _reason(self, state: State) -> dict:
        """Reasoning step using LLM."""
        prompt = f"Reason about: {state.data.get('query')}"
        response = await self._call_llm(prompt)
        return {"thought": response}

    async def _act(self, state: State) -> dict:
        """Execute a tool based on reasoning."""
        return {"action": "stubbed_action"}