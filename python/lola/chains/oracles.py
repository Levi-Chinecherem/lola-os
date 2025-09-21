# Standard imports
import typing as tp

# Local
from .connection import ChainConnection

"""
File: Defines the Oracles class for LOLA OS TMVP 1 Phase 2.

Purpose: Provides interface for fetching oracle data.
How: Uses web3.py to call oracle contracts.
Why: Enables off-chain data access, per EVM-Native tenet.
Full Path: lola-os/python/lola/chains/oracles.py
Future Optimization: Migrate to Rust for fast oracle queries (post-TMVP 1).
"""

class Oracles:
    """Oracles: Fetches data from oracle contracts."""

    def __init__(self, rpc_url: str):
        """
        Initialize with RPC URL.

        Args:
            rpc_url: EVM RPC URL.
        """
        self.connection = ChainConnection(rpc_url)

    async def fetch(self, oracle_address: str, function_name: str, args: tp.List[tp.Any]) -> tp.Any:
        """
        Fetches oracle data.

        Args:
            oracle_address: Oracle contract address.
            function_name: Function to call.
            args: Function arguments.

        Returns:
            Oracle data.

        Does Not: Handle custom oracles—expand in TMVP 2.
        """
        abi = [{"inputs": [], "name": function_name, "outputs": [{"internalType": "uint256", "name": "", "type": "uint256"}], "stateMutability": "view", "type": "function"}]  # Example ABI
        contract = Contract(self.connection, oracle_address, abi)
        return await contract.call_function(function_name, args)

__all__ = ["Oracles"]