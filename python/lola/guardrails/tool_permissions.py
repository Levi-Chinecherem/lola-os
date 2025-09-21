# Standard imports
import typing as tp

# Local
from lola.tools.base import BaseTool

"""
File: Defines the ToolPermissionManager class for LOLA OS TMVP 1 Phase 2.

Purpose: Manages role-based tool permissions.
How: Uses dict for role-tool mapping.
Why: Ensures secure tool access, per Radical Reliability tenet.
Full Path: lola-os/python/lola/guardrails/tool_permissions.py
Future Optimization: Migrate to Rust for secure permissions (post-TMVP 1).
"""

class ToolPermissionManager:
    """ToolPermissionManager: Manages tool permissions. Does NOT persist permissions—use StateManager."""

    def __init__(self):
        """
        Initialize with empty permissions.
        """
        self.permissions = {}

    def add_permission(self, role: str, tool_name: str) -> None:
        """
        Adds permission for a role to use a tool.

        Args:
            role: Role name.
            tool_name: Tool name.
        """
        if role not in self.permissions:
            self.permissions[role] = []
        self.permissions[role].append(tool_name)

    def check_permission(self, role: str, tool: BaseTool) -> bool:
        """
        Checks if a role has permission to use a tool.

        Args:
            role: Role name.
            tool: BaseTool instance.

        Returns:
            True if permitted, False otherwise.
        """
        return tool.name in self.permissions.get(role, [])

__all__ = ["ToolPermissionManager"]