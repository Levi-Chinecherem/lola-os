# Standard imports
from typing import Any, Dict, Optional
import json

# Third-party imports
from web3 import Web3
from web3.types import Address

# Local imports
from lola.tools.base import BaseTool
from lola.utils.logging import logger

"""
File: Contract call tool for EVM read operations in LOLA OS TMVP 1.

Purpose: Enables read-only contract calls on EVM chains.
How: Uses web3.py to interact with smart contracts via provided Web3 instance.
Why: Supports EVM-native agent capabilities, per EVM-Native tenet.
Full Path: lola-os/python/lola/tools/onchain/contract_call.py
"""

class ContractCallTool(BaseTool):
    """Tool for executing read-only EVM contract calls."""

    def __init__(self, web3: Web3, contract_address: Optional[str] = None, abi: Optional[str] = None):
        """
        Initialize the contract call tool.

        Args:
            web3: Web3 instance for EVM interaction.
            contract_address: Optional contract address (hex string).
            abi: Optional contract ABI (JSON string).
        """
        super().__init__()
        self.web3 = web3
        self.contract_address = contract_address
        self.abi = json.loads(abi) if abi else None
        self.contract = None
        if contract_address and abi and self.web3.is_connected():
            self.contract = self.web3.eth.contract(address=contract_address, abi=self.abi)
        logger.info(f"Initialized ContractCallTool with contract: {contract_address}")

    async def execute(self, function_name: str, args: Dict[str, Any] = None) -> Any:
        """
        Execute a read-only contract call.

        Args:
            function_name: Name of the contract function to call.
            args: Optional arguments for the function.

        Returns:
            Result of the contract call.

        Does Not: Perform write operations—TMVP 1 is read-only.
        """
        if not self.contract:
            logger.error("Contract not initialized")
            raise ValueError("Contract not initialized")
        try:
            func = getattr(self.contract.functions, function_name)
            result = func(**(args or {})).call()
            logger.debug(f"Contract call {function_name} result: {result}")
            return result
        except Exception as e:
            logger.error(f"Contract call failed: {str(e)}")
            raise