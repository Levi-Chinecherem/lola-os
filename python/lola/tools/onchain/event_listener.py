# Standard imports
import typing as tp

# Third-party
from web3 import Web3

# Local
from ..base import BaseTool
from lola.chains.connection import ChainConnection

"""
File: Defines the EventListenerTool class for LOLA OS TMVP 1 Phase 2.

Purpose: Enables agents to listen for EVM events.
How: Uses web3.py for real event filtering.
Why: Supports on-chain event monitoring, per EVM-Native tenet.
Full Path: lola-os/python/lola/tools/onchain/event_listener.py
Future Optimization: Migrate to Rust for real-time event handling (post-TMVP 1).
"""

class EventListenerTool(BaseTool):
    """EventListenerTool: Listens for EVM events. Does NOT persist events—use StateManager."""

    name: str = "event_listener"

    def __init__(self, rpc_url: str, address: str, abi: tp.List[dict]):
        """
        Initialize with RPC URL, address, and ABI.

        Args:
            rpc_url: EVM RPC URL.
            address: Contract address.
            abi: Contract ABI.
        """
        connection = ChainConnection(rpc_url)
        self.w3 = connection.w3
        self.contract = self.w3.eth.contract(address=address, abi=abi)

    async def execute(self, input_data: tp.Any) -> tp.Any:
        """
        Listen for an event.

        Args:
            input_data: Dict with 'event_name' and 'from_block'.

        Returns:
            List of event logs.

        Does Not: Handle subscriptions—use for polling.
        """
        if not isinstance(input_data, dict) or 'event_name' not in input_data:
            raise ValueError("Input data must be dict with 'event_name' and 'from_block'.")
        event = getattr(self.contract.events, input_data['event_name'])
        filter_params = {'fromBlock': input_data.get('from_block', 'latest'), 'toBlock': 'latest', 'address': self.contract.address}
        logs = self.w3.eth.get_logs(filter_params)
        return logs

__all__ = ["EventListenerTool"]