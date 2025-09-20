# Standard imports
from typing import List
import asyncio

# Third-party imports
from web3 import Web3

# Local imports
from lola.agents.react import ReActAgent
from lola.tools.onchain.contract_call import ContractCallTool
from lola.utils.logging import logger
from lola.utils.config import load_config

"""
File: Onchain analyst agent example for LOLA OS TMVP 1 Phase 4.

Purpose: Demonstrates a ReAct-based agent for reading NFT floor prices.
How: Uses ContractCallTool to query EVM contracts (read-only).
Why: Showcases EVM-native capabilities, per EVM-Native tenet.
Full Path: lola-os/examples/onchain_analyst/agent.py
"""

class OnchainAnalystAgent(ReActAgent):
    """A ReAct-based agent for analyzing on-chain data like NFT floor prices."""

    def __init__(self, tools: List = None, model: str = "openai/gpt-4o"):
        """
        Initialize the onchain analyst agent.

        Args:
            tools: List of tools (defaults to ContractCallTool).
            model: LLM model name (defaults to openai/gpt-4o).
        """
        config = load_config("examples/onchain_analyst/config.yaml")
        model = config.get("model", model)
        web3 = Web3(Web3.HTTPProvider(config.get("rpc_url", "https://mainnet.infura.io/v3/your-infura-key")))
        tools = tools or [ContractCallTool(web3=web3)]
        super().__init__(tools=tools, model=model)
        logger.info(f"Initialized OnchainAnalystAgent with model {model}")

async def main(query: str):
    """Run the onchain analyst agent with a query."""
    agent = OnchainAnalystAgent()
    result = await agent.run(query)
    logger.info(f"Onchain analysis result: {result.data}")
    return result

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python agent.py <query>")
        sys.exit(1)
    asyncio.run(main(sys.argv[1]))