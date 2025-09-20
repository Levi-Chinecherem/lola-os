# Standard imports
import typing as tp
from abc import ABC, abstractmethod

# Local
from .base import BaseAgent
from lola.core.state import State
from lola.core.graph import StateGraph, Node
from lola.tools.base import BaseTool

"""
File: Defines the PlanExecuteAgent for LOLA OS TMVP 1 Phase 2.

Purpose: Implements a planning and execution agent for structured tasks.
How: Uses StateGraph to generate and execute plans via LLM.
Why: Enables goal-oriented workflows, per Choice by Design.
Full Path: lola-os/python/lola/agents/plan_execute.py
"""
class PlanExecuteAgent(BaseAgent):
    """PlanExecuteAgent: Generates and executes plans. Does NOT persist state—use StateManager."""

    def __init__(self, tools: tp.List[BaseTool], model: str = "openai/gpt-4o"):
        """
        Initialize with tools and LLM model.

        Args:
            tools: List of BaseTool instances.
            model: LLM model string for litellm.
        """
        super().__init__(tools, model)
        self.graph = StateGraph(self.state)
        self.graph.add_node(Node(id="plan", type="llm", function=self._plan, description="Planning step"))
        self.graph.add_node(Node(id="execute", type="tool", function=self._execute, description="Execution step"))

    async def run(self, query: str) -> State:
        """
        Run a plan-execute cycle on the query.

        Args:
            query: User input string.
        Returns:
            State: Updated state with plan/execution.
        Does Not: Persist state—caller must use StateManager.
        """
        self.state.update({"query": query})
        return await self.graph.execute()

    async def _plan(self, state: State) -> dict:
        """Generate a plan using LLM."""
        prompt = f"Generate a plan for: {state.data.get('query')}"
        response = await self._call_llm(prompt)
        return {"plan": response}

    async def _execute(self, state: State) -> dict:
        """Execute the plan using tools."""
        return {"execution": "stubbed_execution"}