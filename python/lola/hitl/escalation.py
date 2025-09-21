# Standard imports
import typing as tp

# Local
from lola.core.state import State
from lola.tools.human_input import HumanInputTool

"""
File: Defines the EscalationHandler class for LOLA OS TMVP 1 Phase 2.

Purpose: Escalates low-confidence tasks to humans.
How: Checks confidence and prompts for input.
Why: Improves reliability in uncertain cases, per Radical Reliability tenet.
Full Path: lola-os/python/lola/hitl/escalation.py
Future Optimization: Migrate to Rust for fast escalation (post-TMVP 1).
"""

class EscalationHandler:
    """EscalationHandler: Handles task escalation to humans. Does NOT persist escalations—use StateManager."""

    def __init__(self, confidence_threshold: float = 0.8):
        """
        Initialize with confidence threshold.

        Args:
            confidence_threshold: Threshold for escalation.
        """
        self.threshold = confidence_threshold
        self.input_tool = HumanInputTool()

    async def escalate(self, state: State, confidence: float) -> State:
        """
        Escalates if confidence is low.

        Args:
            state: Current state.
            confidence: Agent confidence score.

        Returns:
            Updated state with human input if escalated.

        Does Not: Generate confidence—use evals/evaluator.py.
        """
        if confidence < self.threshold:
            prompt = f"Escalate: Low confidence ({confidence}) on {state.query}. Provide guidance."
            guidance = await self.input_tool.execute(prompt)
            state.update(output=guidance)
        return state

__all__ = ["EscalationHandler"]