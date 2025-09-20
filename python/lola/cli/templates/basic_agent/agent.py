# Standard imports
from typing import List

# Local
from lola.agents.react import ReActAgent
from lola.tools.human_input import HumanInputTool
from lola.utils.logging import logger

"""
File: Basic agent template for LOLA OS TMVP 1.

Purpose: Provides a starting point for a ReAct-based agent.
How: Uses ReActAgent with HumanInputTool and configurable model.
Why: Simplifies agent creation, per Developer Sovereignty.
Full Path: lola-os/python/lola/cli/templates/basic_agent/agent.py
"""

class BasicAgent(ReActAgent):
    """A simple ReAct-based agent for LOLA OS."""

    def __init__(self, tools: List, model: str = "openai/gpt-4o"):
        """
        Initialize the basic agent.

        Args:
            tools: List of tools (defaults to HumanInputTool).
            model: LLM model name (defaults to openai/gpt-4o).
        """
        super().__init__(tools=tools or [HumanInputTool()], model=model)
        logger.info(f"Initialized BasicAgent with model {model}")