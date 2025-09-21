# Standard imports
import typing as tp

# Third-party
from web3 import Web3

# Local
from ..base import BaseTool
from lola.chains.contract import Contract

"""
File: Defines the ContractCallTool class for LOLA OS TMVP 1 Phase 2.

Purpose: Enables agents to call EVM contracts.
How: Uses web3.py for real contract calls.
Why: Supports on-chain interactions, per EVM-Native tenet.
Full Path: lola-os/python/lola/tools/onchain/contract_call.py
Future Optimization: Migrate to Rust for low-latency calls (post-TMVP 1).
"""

class ContractCallTool(BaseTool):
    """ContractCallTool: Calls EVM contracts. Does NOT persist results—use StateManager."""

    name: str = "contract_call"

    def __init__(self, rpc_url: str, address: str, abi: tp.List[dict]):
        """
        Initialize with RPC URL, address, and ABI.

        Args:
            rpc_url: EVM RPC URL.
            address: Contract address.
            abi: Contract ABI.
        """
        w3 = Web3(Web3.HTTPProvider(rpc_url))
        self.contract = Contract(w3, address, abi)

    async def execute(self, input_data: tp.Any) -> tp.Any:
        """
        Call a contract function.

        Args:
            input_data: Dict with 'function' and 'args'.

        Returns:
            Call result.

        Does Not: Handle transactions—read-only.
        """
        if not isinstance(input_data, dict) or 'function' not in input_data:
            raise ValueError("Input data must be dict with 'function' and 'args'.")
        return await self.contract.call_function(input_data['function'], input_data.get('args', []))

__all__ = ["ContractCallTool"]