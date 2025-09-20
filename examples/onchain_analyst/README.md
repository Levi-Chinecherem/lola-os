# Onchain Analyst Agent Example

This example demonstrates a ReAct-based agent for reading NFT floor prices using LOLA OS.

## Setup
1. Install LOLA OS:
   ```bash
   poetry install
   ```
2. Set API key and RPC URL in `config.yaml`:
   ```yaml
   model: openai/gpt-4o
   rpc_url: https://mainnet.infura.io/v3/your-infura-key
   api_key: your-api-key-here
   ```
3. Run the agent:
   ```bash
   poetry run python agent.py "Get the floor price of Bored Ape Yacht Club"
   ```

## Features
- Uses `ContractCallTool` for read-only EVM queries via `web3.py`.
- Configurable via `config.yaml` with `litellm` and `web3.py` support.

## Output
The agent returns a `State` object with on-chain data, logged to stdout.
