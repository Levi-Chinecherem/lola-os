# Standard imports
import typing as tp

# Local
from lola.core.graph import Node
from lola.core.state import State
from lola.tools.human_input import HumanInputTool

"""
File: Defines the ApprovalGatewayNode class for LOLA OS TMVP 1 Phase 2.

Purpose: Pauses workflows for human approval.
How: Integrates as a graph node using HumanInputTool for input.
Why: Ensures human oversight in critical steps, per Choice by Design tenet.
Full Path: lola-os/python/lola/hitl/approval.py
Future Optimization: Migrate to Rust for UI integration (post-TMVP 1).
"""

class ApprovalGatewayNode(Node):
    """ApprovalGatewayNode: Pauses for human approval in graphs. Does NOT persist input—use StateManager."""

    def __init__(self, id: str = "approval"):
        """
        Initialize the approval node.

        Args:
            id: Node ID.
        """
        super().__init__(id=id, action=self.approve, type="hitl")
        self.input_tool = HumanInputTool()

    async def approve(self, state: State) -> State:
        """
        Prompts for human approval.

        Args:
            state: Current state.

        Returns:
            Updated state with approval status.

        Does Not: Handle escalation—use escalation.py.
        """
        prompt = f"Approve action? Current output: {state.output}"
        approval = await self.input_tool.execute(prompt)
        state.metadata["approved"] = "yes" in approval.lower()
        return state

__all__ = ["ApprovalGatewayNode"]