"""
File: Initializes the orchestration module for LOLA OS TMVP 1 Phase 2.

Purpose: Exports multi-agent orchestration components.
How: Defines package-level exports for developer imports.
Why: Centralizes access to orchestration utilities, per Choice by Design.
Full Path: lola-os/python/lola/orchestration/__init__.py
"""
from .swarm import AgentSwarmOrchestrator
from .contract_net import ContractNetProtocol
from .blackboard import BlackboardSystem
from .group_chat import GroupChatManager

__all__ = ["AgentSwarmOrchestrator", "ContractNetProtocol", "BlackboardSystem", "GroupChatManager"]