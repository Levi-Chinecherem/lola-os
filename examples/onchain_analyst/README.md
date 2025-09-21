# LOLA OS On-Chain Analyst Example

This example demonstrates a ReAct-based agent that queries Uniswap contract data on the Ethereum testnet (e.g., Sepolia).

## Prerequisites

- Python 3.10+
- Poetry installed
- OpenAI API key
- Access to an Ethereum testnet RPC (e.g., Sepolia)

## Setup

1. Install dependencies:
   ```bash
   poetry install
   ```

2. Set environment variables or update `config.yaml`:
   ```bash
   export OPENAI_API_KEY="your-key"
   export WEB3_PROVIDER_URI="https://sepolia.infura.io/v3/your-infura-key"
   ```

3. Run the agent:
   ```bash
   poetry run python agent.py "Get Uniswap pair price"
   ```

## Expected Output

The agent will query the Uniswap contract and return an answer like:
```
On-chain analysis result: The USDC/WETH pair price is approximately 0.0005 ETH.
```

## Next Steps

- Modify `agent.py` to query different contracts or add tools.
- Explore the LOLA OS documentation for advanced EVM integrations.
