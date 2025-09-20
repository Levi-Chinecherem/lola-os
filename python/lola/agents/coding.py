# Standard imports
import typing as tp
from abc import ABC, abstractmethod

# Local
from .base import BaseAgent
from lola.core.state import State
from lola.core.graph import StateGraph, Node
from lola.tools.base import BaseTool

"""
File: Defines the CodingAgent for LOLA OS TMVP 1 Phase 2.

Purpose: Implements an agent for generating and validating code.
How: Uses StateGraph to manage code generation and validation with LLM.
Why: Enables programmatic tasks, per Developer Sovereignty.
Full Path: lola-os/python/lola/agents/coding.py
"""
class CodingAgent(BaseAgent):
    """CodingAgent: Generates and validates code. Does NOT persist state—use StateManager."""

    def __init__(self, tools: tp.List[BaseTool], model: str = "openai/gpt-4o"):
        """
        Initialize with tools and LLM model.

        Args:
            tools: List of BaseTool instances.
            model: LLM model string for litellm.
        """
        super().__init__(tools, model)
        self.graph = StateGraph(self.state)
        self.graph.add_node(Node(id="code", type="llm", function=self._generate_code, description="Code generation step"))

    async def run(self, query: str) -> State:
        """
        Run a coding cycle on the query.

        Args:
            query: User input string.
        Returns:
            State: Updated state with generated code.
        Does Not: Persist state—caller must use StateManager.
        """
        self.state.update({"query": query})
        return await self.graph.execute()

    async def _generate_code(self, state: State) -> dict:
        """Generate code using LLM."""
        prompt = f"Generate code for: {state.data.get('query')}"
        response = await self._call_llm(prompt)
        return {"code": response}