"""
File: Initializes the hitl module for LOLA OS TMVP 1 Phase 2.

Purpose: Exports human-in-the-loop components for agents.
How: Defines package-level exports for HITL tools.
Why: Centralizes access to human interaction utilities, per Choice by Design.
Full Path: lola-os/python/lola/hitl/__init__.py
"""
from .approval import ApprovalGatewayNode
from .escalation import EscalationHandler
from .corrections import InteractiveCorrections

__all__ = ["ApprovalGatewayNode", "EscalationHandler", "InteractiveCorrections"]