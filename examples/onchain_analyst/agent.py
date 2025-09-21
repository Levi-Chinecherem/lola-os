# Standard imports
import asyncio
from typing import Dict

# Third-party imports
from web3 import Web3

# Local imports
from lola.agents.react import ReActAgent
from lola.tools.onchain.contract_call import ContractCallTool
from lola.utils.config import config, load_config
from lola.utils.logging import logger
from lola.core.state import State

"""
File: On-chain analyst agent example for LOLA OS TMVP 1 Phase 4.

Purpose: Demonstrates a ReAct agent querying Uniswap contract data.
How: Uses ReActAgent with ContractCallTool for real EVM reads.
Why: Showcases EVM-native agent capabilities, per EVM-Native tenet.
Full Path: lola-os/examples/onchain_analyst/agent.py
"""

async def main(query: str) -> str:
    """
    Run an on-chain analyst agent to query Uniswap contract data.

    Args:
        query: Input query (e.g., "Get Uniswap pair price").

    Returns:
        Final answer as a string.

    Does not:
        Perform EVM writes; only read operations.
    """
    # Load configuration
    config_path = "config.yaml"
    load_config(config_path)

    # Initialize Web3 provider
    w3 = Web3(Web3.HTTPProvider(config.get("web3_provider_uri")))
    if not w3.is_connected():
        logger.error("Failed to connect to Web3 provider.")
        raise RuntimeError("Web3 connection failed")

    # Uniswap V2 pair contract (example: USDC/WETH on Sepolia)
    contract_address = "0x1c3140aB59d6cAf9fa7459C6fBC1F1B7c3e7c4B0"  # Replace with actual address
    abi = [
        {
            "constant": True,
            "inputs": [],
            "name": "getReserves",
            "outputs": [
                {"name": "_reserve0", "type": "uint112"},
                {"name": "_reserve1", "type": "uint112"},
                {"name": "_blockTimestampLast", "type": "uint32"},
            ],
            "type": "function",
        }
    ]

    # Initialize tools
    contract_tool = ContractCallTool(w3=w3, contract_address=contract_address, abi=abi)
    
    # Initialize agent
    agent = ReActAgent(
        model=config.get("openai_api_key"),
        tools=[contract_tool]
    )
    
    # Run agent
    logger.info(f"Processing query: {query}")
    state = await agent.run(query)
    
    return state.output

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        logger.error("Please provide a query as an argument.")
        sys.exit(1)
    
    result = asyncio.run(main(sys.argv[1]))
    logger.info(f"On-chain analysis result: {result}")