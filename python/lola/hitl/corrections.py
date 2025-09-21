# Standard imports
import typing as tp

# Local
from lola.core.state import State
from lola.tools.human_input import HumanInputTool

"""
File: Defines the InteractiveCorrections class for LOLA OS TMVP 1 Phase 2.

Purpose: Allows human corrections in agent workflows.
How: Prompts for corrections and updates state.
Why: Enables error recovery, per Choice by Design tenet.
Full Path: lola-os/python/lola/hitl/corrections.py
Future Optimization: Migrate to Rust for interactive UI (post-TMVP 1).
"""

class InteractiveCorrections:
    """InteractiveCorrections: Handles human corrections. Does NOT persist corrections—use StateManager."""

    def __init__(self):
        """Initialize the corrections handler."""
        self.input_tool = HumanInputTool()

    async def correct(self, state: State, error: str) -> State:
        """
        Prompts for correction on error.

        Args:
            state: Current state.
            error: Error description.

        Returns:
            Updated state with correction.

        Does Not: Handle automatic corrections—use meta_cognition.py.
        """
        prompt = f"Correct error: {error}\nCurrent output: {state.output}"
        correction = await self.input_tool.execute(prompt)
        state.update(output=correction)
        return state

__all__ = ["InteractiveCorrections"]