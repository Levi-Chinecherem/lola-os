# Standard imports
import typing as tp

# Third-party
from pydantic import BaseModel

# Local
from lola.core.agent import BaseAgent
from lola.core.state import State
from lola.tools.base import BaseTool

"""
File: Defines the base class for all agent templates in LOLA OS TMVP 1 Phase 2.

Purpose: Provides a foundation for specialized agents with tool and state management.
How: Extends core BaseAgent with tool binding and basic execution logic.
Why: Ensures consistent agent interfaces while allowing customization, per Developer Sovereignty and Explicit over Implicit tenets.
Full Path: lola-os/python/lola/agents/base.py
Future Optimization: Migrate to Rust for concurrent agent execution (post-TMVP 1).
"""

class BaseTemplateAgent(BaseAgent):
    """Abstract base class for agent templates. Does NOT persist state—use StateManager."""

    def __init__(self, tools: tp.List[BaseTool], model: str = "openai/gpt-4o"):
        """
        Initialize the agent with tools and LLM model.

        Args:
            tools: List of BaseTool instances for agent actions.
            model: LLM model string for litellm (e.g., "openai/gpt-4o", "anthropic/claude-3-sonnet").
        """
        super().__init__(tools, model)
        # Inline: Why list comprehension? To create a dict for fast tool lookup by name.
        self.tools_dict = {tool.name: tool for tool in tools}

    async def run(self, query: str) -> State:
        """
        Execute the agent's logic for a given query.

        Args:
            query: User input string.
        Returns:
            State: Updated agent state after execution.
        Does Not: Persist state—caller must use StateManager.
        """
        # Inline: Subclasses override this for specific patterns (e.g., ReAct loop).
        raise NotImplementedError("Subclasses must implement run method.")

    async def execute_tool(self, tool_name: str, params: tp.Dict[str, tp.Any]) -> tp.Any:
        """
        Execute a tool by name with parameters.

        Args:
            tool_name: Name of the tool to execute.
            params: Dictionary of parameters for the tool.
        Returns:
            tp.Any: Result of tool execution.
        """
        tool = self.tools_dict.get(tool_name)
        if not tool:
            raise ValueError(f"Tool {tool_name} not found.")
        # Inline: Why await? Tools may be async (e.g., network calls).
        return await tool.execute(**params)

__all__ = ["BaseTemplateAgent"]