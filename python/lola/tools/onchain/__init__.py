"""
File: Initializes the onchain tools module for LOLA OS TMVP 1 Phase 2.

Purpose: Exports EVM read-only tools for agents.
How: Defines package-level exports for onchain tools.
Why: Centralizes access to EVM tools, per Developer Sovereignty.
Full Path: lola-os/python/lola/tools/onchain/__init__.py
"""

from .contract_call import ContractCallTool
from .event_listener import EventListenerTool
from .oracle_fetch import OracleFetchTool

__all__ = [
    "ContractCallTool",
    "EventListenerTool",
    "OracleFetchTool",
]