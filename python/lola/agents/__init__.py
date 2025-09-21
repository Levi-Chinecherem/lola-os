"""
File: Initializes the agents module for LOLA OS TMVP 1 Phase 2.

Purpose: Exports agent templates for various use cases.
How: Defines package-level exports for developer imports.
Why: Centralizes access to agent implementations, per Developer Sovereignty.
Full Path: lola-os/python/lola/agents/__init__.py
"""

from .base import BaseAgent
from .react import ReActAgent
from .plan_execute import PlanExecuteAgent
from .collaborative import CollaborativeAgent
from .router import RouterAgent
from .conversational import ConversationalAgent
from .coding import CodingAgent
from .orchestrator import OrchestratorAgent
from .meta_cognition import MetaCognitionAgent
from .synthetic_data import SyntheticDataAgent
from .legacy_interface import LegacyInterfaceAgent

__all__ = [
    "BaseAgent",
    "ReActAgent",
    "PlanExecuteAgent",
    "CollaborativeAgent",
    "RouterAgent",
    "ConversationalAgent",
    "CodingAgent",
    "OrchestratorAgent",
    "MetaCognitionAgent",
    "SyntheticDataAgent",
    "LegacyInterfaceAgent",
]