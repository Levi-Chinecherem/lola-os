# Standard imports
import typing as tp

# Local
from .base import BaseTool

"""
File: Defines the HumanInputTool class for LOLA OS TMVP 1 Phase 2.

Purpose: Enables agents to pause for human input.
How: Uses input() for real user interaction.
Why: Supports human-in-the-loop workflows, per Choice by Design tenet.
Full Path: lola-os/python/lola/tools/human_input.py
Future Optimization: Migrate to Rust for UI-based input (post-TMVP 1).
"""

class HumanInputTool(BaseTool):
    """HumanInputTool: Pauses for human input. Does NOT persist input—use StateManager."""

    name: str = "human_input"

    async def execute(self, input_data: tp.Any) -> tp.Any:
        """
        Prompt for human input.

        Args:
            input_data: Prompt string for the user.

        Returns:
            User input string.

        Does Not: Handle timeouts—use hitl/escalation.py.
        """
        if not isinstance(input_data, str):
            raise ValueError("Input data must be a prompt string.")
        return input(input_data + ": ")

__all__ = ["HumanInputTool"]