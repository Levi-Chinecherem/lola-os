# Standard imports
import typing as tp

# Local
from .connection import ChainConnection

"""
File: Defines the Contract class for LOLA OS TMVP 1 Phase 2.

Purpose: Provides an interface for interacting with EVM contracts.
How: Wraps web3.py contract calls.
Why: Simplifies contract operations for tools, per EVM-Native tenet.
Full Path: lola-os/python/lola/chains/contract.py
Future Optimization: Migrate to Rust for efficient calls (post-TMVP 1).
"""

class Contract:
    """Contract: Manages EVM contract interactions."""

    def __init__(self, connection: ChainConnection, address: str, abi: tp.List[dict]):
        """
        Initialize with connection, address, and ABI.

        Args:
            connection: ChainConnection instance.
            address: Contract address.
            abi: Contract ABI.
        """
        self.contract = connection.w3.eth.contract(address=address, abi=abi)

    async def call_function(self, function_name: str, args: tp.List[tp.Any]) -> tp.Any:
        """
        Calls a contract function.

        Args:
            function_name: Function name.
            args: Function arguments.

        Returns:
            Function result.

        Does Not: Handle transactions—read-only.
        """
        func = getattr(self.contract.functions, function_name)
        return func(*args).call()

__all__ = ["Contract"]