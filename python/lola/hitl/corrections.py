# Standard imports
import typing as tp

# Local
from lola.tools.human_input import HumanInputTool
from lola.core.state import State

"""
File: Defines the InteractiveCorrections for LOLA OS TMVP 1 Phase 2.

Purpose: Allows humans to correct agent mistakes.
How: Uses HumanInputTool to update State.
Why: Enables error correction, per Choice by Design.
Full Path: lola-os/python/lola/hitl/corrections.py
"""
class InteractiveCorrections:
    """InteractiveCorrections: Handles human corrections. Does NOT persist state—use StateManager."""

    def __init__(self, human_input_tool: HumanInputTool):
        """
        Initialize with a human input tool.

        Args:
            human_input_tool: HumanInputTool instance.
        """
        self.human_input_tool = human_input_tool

    async def correct(self, state: State, error: str) -> dict:
        """
        Correct an agent mistake.

        Args:
            state: Current state.
            error: Error description.
        Returns:
            dict: Correction result (stubbed for now).
        """
        return self.human_input_tool.execute(f"Correct error: {error}")