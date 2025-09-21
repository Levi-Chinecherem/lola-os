# Standard imports
import typing as tp

# Third-party
from web3 import Web3

# Local
from ..base import BaseTool
from lola.chains.oracles import Oracles

"""
File: Defines the OracleFetchTool class for LOLA OS TMVP 1 Phase 2.

Purpose: Enables agents to fetch data from on-chain oracles.
How: Uses web3.py to call oracle contracts.
Why: Supports off-chain data access, per EVM-Native tenet.
Full Path: lola-os/python/lola/tools/onchain/oracle_fetch.py
Future Optimization: Migrate to Rust for fast oracle queries (post-TMVP 1).
"""

class OracleFetchTool(BaseTool):
    """OracleFetchTool: Fetches data from oracles. Does NOT persist data—use StateManager."""

    name: str = "oracle_fetch"

    def __init__(self, rpc_url: str):
        """
        Initialize with RPC URL.

        Args:
            rpc_url: EVM RPC URL.
        """
        self.oracles = Oracles(rpc_url)

    async def execute(self, input_data: tp.Any) -> tp.Any:
        """
        Fetch oracle data.

        Args:
            input_data: Dict with 'oracle_type' and 'params'.

        Returns:
            Oracle data.

        Does Not: Handle custom oracles—use chains/oracles.py.
        """
        if not isinstance(input_data, dict) or 'oracle_type' not in input_data:
            raise ValueError("Input data must be dict with 'oracle_type' and 'params'.")
        return await self.oracles.fetch(input_data['oracle_type'], input_data.get('params', {}))

__all__ = ["OracleFetchTool"]