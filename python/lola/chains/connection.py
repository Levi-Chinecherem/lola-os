# Standard imports
import typing as tp

# Third-party
from web3 import Web3

"""
File: Defines the ChainConnection class for LOLA OS TMVP 1 Phase 2.

Purpose: Provides a connection to EVM chains for read operations.
How: Uses web3.py to connect to RPC providers.
Why: Enables real EVM access for agents, per EVM-Native tenet.
Full Path: lola-os/python/lola/chains/connection.py
Future Optimization: Migrate to Rust for fast connections (post-TMVP 1).
"""

class ChainConnection:
    """ChainConnection: Manages connection to EVM chains."""

    def __init__(self, rpc_url: str):
        """
        Initialize with RPC URL.

        Args:
            rpc_url: EVM RPC URL.
        """
        self.w3 = Web3(Web3.HTTPProvider(rpc_url))
        if not self.w3.is_connected():
            raise ConnectionError("Failed to connect to EVM chain.")

    def get_block_number(self) -> int:
        """
        Gets the current block number.

        Returns:
            Current block number.

        Does Not: Handle errors—caller must wrap in try/except.
        """
        return self.w3.eth.block_number

__all__ = ["ChainConnection"]