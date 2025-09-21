# Standard imports
import typing as tp
import logging

# Third-party
from web3 import Web3
from tenacity import retry, stop_after_attempt, wait_exponential

# Local
from lola.chains.connection import ChainConnection
from lola.utils import sentry

"""
File: Implements Web3ContractAdapter for LOLA OS TMVP 1 Phase 5.

Purpose: Wraps web3.py contract interactions with production readiness.
How: Configures contract with address and ABI.
Why: Simplifies contract calls, per EVM-Native and Radical Reliability tenets.
Full Path: lola-os/python/lola/libs/web3/contract.py
"""

logger = logging.getLogger(__name__)

class Web3ContractAdapter:
    """Web3ContractAdapter: Wraps web3.py contract interactions."""

    def __init__(self, connection: ChainConnection, address: str, abi: tp.List[dict]):
        """
        Initialize with connection, address, and ABI.

        Args:
            connection: ChainConnection instance.
            address: Contract address.
            abi: Contract ABI.

        Does Not: Handle transactions—read-only.
        """
        try:
            self.contract = connection.w3.eth.contract(address=address, abi=abi)
            logger.info(f"Web3ContractAdapter initialized for {address}")
        except Exception as e:
            logger.error(f"Web3ContractAdapter initialization failed: {e}")
            sentry.capture_exception(e)
            raise

__all__ = ["Web3ContractAdapter"]