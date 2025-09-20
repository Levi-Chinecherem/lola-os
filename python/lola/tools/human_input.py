# Standard imports
import typing as tp

# Local
from .base import BaseTool

"""
File: Defines the HumanInputTool for LOLA OS TMVP 1 Phase 2.

Purpose: Provides a tool for pausing and requesting human input.
How: Executes a stubbed input request (to be extended with CLI/GUI).
Why: Enables human-in-the-loop workflows, per Choice by Design.
Full Path: lola-os/python/lola/tools/human_input.py
"""
class HumanInputTool(BaseTool):
    """HumanInputTool: Requests human input. Does NOT persist input—use StateManager."""

    name: str = "human_input"

    def execute(self, *args, **kwargs) -> dict:
        """
        Request human input.

        Args:
            *args: Prompt string as first positional argument.
            **kwargs: Optional parameters (e.g., timeout).
        Returns:
            dict: Human input (stubbed for now).
        """
        prompt = args[0] if args else kwargs.get("prompt", "")
        return {"input": f"Stubbed human input for: {prompt}"}