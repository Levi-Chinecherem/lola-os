# Standard imports
import typing as tp
import ast

# Local
from .base import BaseTemplateAgent
from lola.core.state import State
from lola.tools.code_interpreter import CodeInterpreterTool
from lola.tools.base import BaseTool

"""
File: Defines the CodingAgent class for LOLA OS TMVP 1 Phase 2.

Purpose: Implements an agent for code generation and execution.
How: Uses LLM to generate code, then executes with CodeInterpreterTool.
Why: Enables programmatic tasks, per Developer Sovereignty tenet.
Full Path: lola-os/python/lola/agents/coding.py
Future Optimization: Migrate to Rust for secure code sandboxing (post-TMVP 1).
"""

class CodingAgent(BaseTemplateAgent):
    """CodingAgent: Implements code generation and execution pattern. Does NOT persist state—use StateManager."""

    def __init__(self, tools: tp.List[BaseTool], model: str = "openai/gpt-4o"):
        """
        Initialize with tools and LLM model.

        Args:
            tools: List of BaseTool instances, including CodeInterpreterTool.
            model: LLM model string for litellm.
        """
        super().__init__(tools, model)
        self.code_tool = next((tool for tool in tools if isinstance(tool, CodeInterpreterTool)), None)
        if not self.code_tool:
            raise ValueError("CodeInterpreterTool required for CodingAgent.")

    async def run(self, query: str) -> State:
        """
        Generate and execute code for the query.

        Args:
            query: User input string (e.g., "Write a function to add numbers").
        Returns:
            State: Updated state with code and execution result.
        Does Not: Handle code validation—use guardrails/content_safety.py.
        """
        self.state.update({"query": query})
        # Inline: Generate code with LLM
        code_prompt = f"Generate Python code for: {query}"
        code = await self._call_llm(code_prompt)
        self.state.update({"code": code})
        # Inline: Execute generated code
        try:
            result = await self.execute_tool(self.code_tool, code)
            self.state.update({"output": result})
        except Exception as e:
            self.state.update({"output": str(e)})
        return self.state

__all__ = ["CodingAgent"]