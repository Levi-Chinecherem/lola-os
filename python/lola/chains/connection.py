# Standard imports
import typing as tp

# Third-party
from web3 import Web3

"""
File: Defines the ChainConnection for LOLA OS TMVP 1 Phase 2.

Purpose: Provides a unified connection to EVM chains.
How: Uses web3.py to initialize a Web3 provider.
Why: Enables read-only chain interactions, per EVM-Native tenet.
Full Path: lola-os/python/lola/chains/connection.py
"""
class ChainConnection:
    """ChainConnection: Manages EVM chain connections. Does NOT handle writes—TMVP 2."""

    def __init__(self, provider_url: str):
        """
        Initialize with a provider URL.

        Args:
            provider_url: URL of the EVM provider (e.g., Infura).
        """
        self.web3 = Web3(Web3.HTTPProvider(provider_url))

    def is_connected(self) -> bool:
        """
        Check if the connection is active.

        Returns:
            bool: True if connected, False otherwise.
        """
        return self.web3.is_connected()