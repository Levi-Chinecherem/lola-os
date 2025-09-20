# Standard imports
import typing as tp

# Local
from lola.core.graph import Node
from lola.tools.human_input import HumanInputTool

"""
File: Defines the ApprovalGatewayNode for LOLA OS TMVP 1 Phase 2.

Purpose: Pauses workflows for human approval.
How: Uses HumanInputTool to request approval via Node.
Why: Enables human oversight, per Choice by Design.
Full Path: lola-os/python/lola/hitl/approval.py
"""
class ApprovalGatewayNode(Node):
    """ApprovalGatewayNode: Pauses for human approval. Does NOT persist state—use StateManager."""

    def __init__(self, human_input_tool: HumanInputTool, id: str = "approval"):
        """
        Initialize with a human input tool.

        Args:
            human_input_tool: HumanInputTool instance.
            id: Node identifier.
        """
        super().__init__(id=id, type="hitl", function=self._approve, description="Human approval step")
        self.human_input_tool = human_input_tool

    async def _approve(self, state: tp.Any) -> dict:
        """
        Request human approval.

        Args:
            state: Current state.
        Returns:
            dict: Approval result (stubbed for now).
        """
        return self.human_input_tool.execute("Request approval")