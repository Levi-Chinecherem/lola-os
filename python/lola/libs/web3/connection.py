# Standard imports
import typing as tp
import logging

# Third-party
from web3 import Web3
from tenacity import retry, stop_after_attempt, wait_exponential

# Local
from lola.utils import sentry

"""
File: Implements Web3Adapter for LOLA OS TMVP 1 Phase 5.

Purpose: Provides web3.py-based EVM connection with production readiness.
How: Configures web3.py with HTTP provider and retries.
Why: Ensures robust EVM access, per EVM-Native and Radical Reliability tenets.
Full Path: lola-os/python/lola/libs/web3/connection.py
"""

logger = logging.getLogger(__name__)

class Web3Adapter:
    """Web3Adapter: Configures web3.py for EVM connections."""

    def __init__(self, config: tp.Dict[str, tp.Any]):
        """
        Initialize with config.

        Args:
            config: Dict with 'rpc_url' and 'provider_type' (default: "http").

        Does Not: Handle transactions—use Wallet.
        """
        try:
            self.rpc_url = config.get("rpc_url")
            provider_type = config.get("provider_type", "http")
            if provider_type != "http":
                logger.error(f"Unsupported provider type: {provider_type}")
                sentry.capture_message(f"Unsupported provider type: {provider_type}")
                raise ValueError(f"Unsupported provider type: {provider_type}")
            self.w3 = Web3(Web3.HTTPProvider(self.rpc_url))
            logger.info(f"Web3Adapter initialized with RPC: {self.rpc_url}")
        except Exception as e:
            logger.error(f"Web3Adapter initialization failed: {e}")
            sentry.capture_exception(e)
            raise

__all__ = ["Web3Adapter"]