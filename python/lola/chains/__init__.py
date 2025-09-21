"""
File: Initializes the chains module for LOLA OS TMVP 1 Phase 2.

Purpose: Exports EVM read-only abstractions for agents.
How: Defines package-level exports for chain-related components.
Why: Centralizes access to EVM utilities, per EVM-Native tenet.
Full Path: lola-os/python/lola/chains/__init__.py
"""

from .connection import ChainConnection
from .contract import Contract
from .wallet import Wallet
from .key_manager import KeyManager
from .oracles import Oracles
from .utils import Utils

__all__ = [
    "ChainConnection",
    "Contract",
    "Wallet",
    "KeyManager",
    "Oracles",
    "Utils",
]