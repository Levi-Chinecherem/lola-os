# Standard imports
import typing as tp

# Third-party
from web3 import Web3

# Local
from ...chains.connection import ChainConnection
from ..base import BaseTool

"""
File: Defines the EventListenerTool for LOLA OS TMVP 1 Phase 2.

Purpose: Provides a tool for listening to EVM events.
How: Uses web3.py to monitor contract events via ChainConnection.
Why: Enables agents to track on-chain events, per EVM-Native tenet.
Full Path: lola-os/python/lola/tools/onchain/event_listener.py
"""
class EventListenerTool(BaseTool):
    """EventListenerTool: Monitors EVM events. Does NOT handle writes—TMVP 2."""

    name: str = "event_listener"

    def __init__(self, connection: ChainConnection):
        """
        Initialize with a ChainConnection instance.

        Args:
            connection: ChainConnection object for EVM interactions.
        """
        self.connection = connection

    def execute(self, *args, **kwargs) -> dict:
        """
        Monitor an EVM event.

        Args:
            *args: Event name as first positional argument.
            **kwargs: Optional parameters (e.g., contract_address).
        Returns:
            dict: Event monitoring results (stubbed for now).
        """
        event = args[0] if args else kwargs.get("event", "")
        return {"results": f"Stubbed event monitoring for: {event}"}