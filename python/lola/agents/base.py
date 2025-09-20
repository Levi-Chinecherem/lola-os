# Standard imports
import typing as tp
from abc import ABC

# Local
from lola.core.agent import BaseAgent as CoreBaseAgent
from lola.core.state import State
from lola.tools.base import BaseTool

"""
File: Extends BaseAgent for agent templates in LOLA OS TMVP 1 Phase 2.

Purpose: Provides a common base for all agent implementations with shared utilities.
How: Extends core BaseAgent with template-specific initialization.
Why: Ensures consistent agent interfaces, per Developer Sovereignty.
Full Path: lola-os/python/lola/agents/base.py
"""
class BaseAgent(CoreBaseAgent, ABC):
    """Base class for agent templates, extending core BaseAgent."""

    def __init__(self, tools: tp.List[BaseTool], model: str = "openai/gpt-4o"):
        """
        Initialize the agent with tools and LLM model.

        Args:
            tools: List of BaseTool instances.
            model: LLM model string for litellm.
        """
        super().__init__(tools, model)