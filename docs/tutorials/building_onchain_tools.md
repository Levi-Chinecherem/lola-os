# Building On-Chain Tools

This tutorial guides you through creating an on-chain tool for LOLA OS.

## Step 1: Scaffold a Project

```bash
poetry run lola create my_onchain --template react
cd my_onchain
```

## Step 2: Configure the Agent

Edit `config.yaml`:
```yaml
openai_api_key: "your-openai-api-key"
web3_provider_uri: "https://sepolia.infura.io/v3/your-infura-key"
```

## Step 3: Create a Contract Tool

Create `agents/contract_tool.py`:
```python
from lola.tools.onchain.contract_call import ContractCallTool
from web3 import Web3

w3 = Web3(Web3.HTTPProvider(config.get("web3_provider_uri")))
contract_address = "0x1c3140aB59d6cAf9fa7459C6fBC1F1B7c3e7c4B0"
abi = [{"constant": True, "inputs": [], "name": "getReserves", "outputs": [{"name": "_reserve0", "type": "uint112"}, {"name": "_reserve1", "type": "uint112"}, {"name": "_blockTimestampLast", "type": "uint32"}], "type": "function"}]
tool = ContractCallTool(w3=w3, contract_address=contract_address, abi=abi)
```

## Step 4: Run the Agent

Edit `agents/main.py` to use the tool, then run:
```bash
poetry run lola run agents/main.py "Get Uniswap pair price"
```

## Expected Output

```
Agent output: The USDC/WETH pair price is approximately 0.0005 ETH.
```

## Next Steps

- Explore other EVM tools in `lola.tools.onchain`.
- Run the on-chain analyst example: [On-Chain Analyst](../../examples/onchain_analyst/README.md).
