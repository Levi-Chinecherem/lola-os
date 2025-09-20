# Standard imports
import typing as tp
from abc import ABC, abstractmethod

# Local
from .base import BaseAgent
from lola.core.state import State
from lola.core.graph import StateGraph, Node
from lola.tools.base import BaseTool

"""
File: Defines the ConversationalAgent for LOLA OS TMVP 1 Phase 2.

Purpose: Implements a chatbot-style conversational agent.
How: Uses StateGraph to manage dialog flows with LLM.
Why: Enables interactive user experiences, per Developer Sovereignty.
Full Path: lola-os/python/lola/agents/conversational.py
"""
class ConversationalAgent(BaseAgent):
    """ConversationalAgent: Handles dialog flows. Does NOT persist state—use StateManager."""

    def __init__(self, tools: tp.List[BaseTool], model: str = "openai/gpt-4o"):
        """
        Initialize with tools and LLM model.

        Args:
            tools: List of BaseTool instances.
            model: LLM model string for litellm.
        """
        super().__init__(tools, model)
        self.graph = StateGraph(self.state)
        self.graph.add_node(Node(id="converse", type="llm", function=self._converse, description="Conversation step"))

    async def run(self, query: str) -> State:
        """
        Run a conversational cycle on the query.

        Args:
            query: User input string.
        Returns:
            State: Updated state with conversation results.
        Does Not: Persist state—caller must use StateManager.
        """
        self.state.update({"query": query})
        return await self.graph.execute()

    async def _converse(self, state: State) -> dict:
        """Handle conversation using LLM."""
        prompt = f"Respond conversationally to: {state.data.get('query')}"
        response = await self._call_llm(prompt)
        return {"response": response}