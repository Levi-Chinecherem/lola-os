import abc
import typing as tp
from pydantic import BaseModel
import litellm
from .graph import StateGraph
from .state import State

"""
File: Defines the BaseAgent abstract class for LOLA OS agent implementations.

Purpose: Provides a standardized interface for all agents to execute tasks using a graph-based workflow and LLM integration.
How: Initializes with a StateGraph, uses litellm for LLM calls, and defines abstract methods for task execution.
Why: Ensures developer sovereignty by allowing custom agents while maintaining consistent orchestration, per LOLA's tenets.
Full Path: lola-os/python/lola/core/agent.py
"""

class BaseAgent(abc.ABC):
    """BaseAgent: Abstract base class for all LOLA agents. Does NOT handle state persistence—use StateManager."""

    def __init__(self, model: str, tools: tp.List[tp.Any] = None, graph: tp.Optional[StateGraph] = None):
        """
        Initialize BaseAgent with a model, tools, and optional graph.

        Args:
            model: LLM model string for litellm (e.g., "openai/gpt-4o").
            tools: List of tools the agent can use (expanded in Phase 2).
            graph: Optional StateGraph for workflow; created if None.

        Does Not: Persist state or connect to EVM—handled by StateManager and chains.
        """
        self.model = model
        self.tools = tools or []
        self.graph = graph or StateGraph(State)
        # Inline: Initialize empty state for new sessions
        self.state = State()

    def call_llm(self, prompt: str) -> str:
        """
        Calls the LLM via litellm with the configured model.

        Args:
            prompt: Input prompt string.

        Returns:
            LLM response as a string.

        Does Not: Handle retries or fallbacks—use agnostic/fallback.py in Phase 2.
        """
        # Inline: Use litellm for unified LLM access
        response = litellm.completion(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1000
        )
        return response.choices[0].message.content

    @abc.abstractmethod
    def run(self, query: str) -> State:
        """
        Abstract method to execute the agent's task.

        Args:
            query: User input string.

        Returns:
            Updated State with results.

        Does Not: Handle errors—caller must wrap in try/except.
        """
        pass

    def bind_tools(self, tools: tp.List[tp.Any]) -> None:
        """
        Binds tools to the agent.

        Args:
            tools: List of tools to bind.

        Does Not: Validate tool permissions—use guardrails/tool_permissions.py in Phase 2.
        """
        self.tools.extend(tools)

__all__ = ["BaseAgent"]