# Standard imports
import typing as tp

# Third-party
from eth_account import Account

# Local
from .connection import ChainConnection

"""
File: Defines the Wallet class for LOLA OS TMVP 1 Phase 2.

Purpose: Manages EVM wallet for read operations (balance checks).
How: Uses eth_account for wallet management.
Why: Enables agent wallet interactions, per EVM-Native tenet.
Full Path: lola-os/python/lola/chains/wallet.py
Future Optimization: Migrate to Rust for secure wallet handling (post-TMVP 1).
"""

class Wallet:
    """Wallet: Manages EVM wallet operations."""

    def __init__(self, connection: ChainConnection, private_key: str):
        """
        Initialize with connection and private key.

        Args:
            connection: ChainConnection instance.
            private_key: Wallet private key.
        """
        self.connection = connection
        self.account = Account.from_key(private_key)

    async def get_balance(self) -> int:
        """
        Gets the wallet balance.

        Returns:
            Balance in wei.

        Does Not: Handle transactions—read-only.
        """
        return self.connection.w3.eth.get_balance(self.account.address)

__all__ = ["Wallet"]