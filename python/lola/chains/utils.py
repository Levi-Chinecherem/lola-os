# Standard imports
import typing as tp

# Local
from .connection import ChainConnection

"""
File: Defines the utils module for LOLA OS TMVP 1 Phase 2.

Purpose: Provides helper functions for EVM operations.
How: Implements gas estimation and transaction building.
Why: Simplifies EVM interactions, per EVM-Native tenet.
Full Path: lola-os/python/lola/chains/utils.py
Future Optimization: Migrate to Rust for efficient utils (post-TMVP 1).
"""

class Utils:
    """Utils: EVM helper functions."""

    def __init__(self, connection: ChainConnection):
        """
        Initialize with connection.

        Args:
            connection: ChainConnection instance.
        """
        self.connection = connection

    async def estimate_gas(self, tx: tp.Dict[str, tp.Any]) -> int:
        """
        Estimates gas for a transaction.

        Args:
            tx: Transaction dict.

        Returns:
            Estimated gas.

        Does Not: Execute transaction—read-only.
        """
        return self.connection.w3.eth.estimate_gas(tx)

__all__ = ["Utils"]