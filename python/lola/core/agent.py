# Standard imports
import typing as tp
from abc import ABC, abstractmethod

# Third-party
from pydantic import BaseModel
import litellm

# Local
from .state import State
from ..tools.base import BaseTool

"""
File: Defines the BaseAgent abstract class for LOLA OS TMVP 1.

Purpose: Provides the foundation for all agent implementations (e.g., ReAct, PlanExecute).
How: Defines abstract methods for running agents with state and tool integration, using litellm for LLM calls.
Why: Ensures consistent agent interfaces, per Developer Sovereignty and Explicit over Implicit tenets.
Full Path: lola-os/python/lola/core/agent.py
Future Optimization: Migrate to Rust for concurrent agent execution (post-TMVP 1).
"""

class BaseAgent(ABC):
    """Abstract base class for all LOLA OS agents. Does NOT persist state—use StateManager."""

    def __init__(self, tools: tp.List[BaseTool], model: str = "openai/gpt-4o"):
        """
        Initialize the agent with tools and LLM model.

        Args:
            tools: List of BaseTool instances for agent actions.
            model: LLM model string for litellm (e.g., "openai/gpt-4o", "anthropic/claude-3-sonnet").
        """
        self.tools = {tool.name: tool for tool in tools}
        self.model = model  # Unified LLM string for litellm
        self.state = State()  # Initialize empty state

    @abstractmethod
    async def run(self, query: str) -> State:
        """
        Execute the agent's logic for a given query.

        Args:
            query: User input string.
        Returns:
            State: Updated agent state after execution.
        Does Not: Persist state—caller must use StateManager.
        """
        pass

    async def _call_llm(self, prompt: str) -> str:
        """
        Helper to call LLM via litellm.

        Args:
            prompt: Input prompt string.
        Returns:
            str: LLM response.
        """
        # Inline: Why litellm? Unified interface for model switching per Agnostic Adapter Pattern.
        response = await litellm.acompletion(
            model=self.model,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content