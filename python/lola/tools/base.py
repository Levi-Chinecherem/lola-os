# Standard imports
import typing as tp
from abc import ABC, abstractmethod

# Local
from lola.core.state import State

"""
File: Defines the BaseTool abstract class for LOLA OS TMVP 1 Phase 2.

Purpose: Provides a foundation for all tool implementations with an execute interface.
How: Defines abstract execute method for async tool operations.
Why: Ensures consistent tool usage in agents, per Developer Sovereignty and Explicit over Implicit tenets.
Full Path: lola-os/python/lola/tools/base.py
Future Optimization: Migrate to Rust for high-performance tool execution (post-TMVP 1).
"""

class BaseTool(ABC):
    """Abstract base class for all LOLA OS tools. Does NOT persist state—use StateManager."""

    name: str = "base_tool"

    @abstractmethod
    async def execute(self, input_data: tp.Any) -> tp.Any:
        """
        Execute the tool with input data.

        Args:
            input_data: Input for the tool (e.g., query string or dict).

        Returns:
            tp.Any: Tool result (e.g., str, dict).

        Does Not: Handle validation—caller must validate input.
        """
        pass

__all__ = [
    "BaseTool",
]