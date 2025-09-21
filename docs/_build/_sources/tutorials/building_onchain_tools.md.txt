# Tutorial: Building Onchain Tools

This tutorial shows how to build an EVM read tool for LOLA OS agents.

## Step 1: Create a Project
```bash
poetry run lola create onchain_tool
cd onchain_tool
poetry install
```

## Step 2: Configure the Agent
Edit `config.yaml`:
```yaml
model: openai/gpt-4o
rpc_url: https://mainnet.infura.io/v3/your-infura-key
api_key: your-api-key-here
```

## Step 3: Create a Custom Tool
Create `tools/custom_contract.py`:
```python
from lola.tools.onchain.contract_call import ContractCallTool
from lola.utils.logging import logger
from web3 import Web3

class CustomContractTool(ContractCallTool):
    def __init__(self, web3: Web3):
        super().__init__(web3=web3)
        logger.info("Initialized CustomContractTool")
```

## Step 4: Update the Agent
Modify `agent.py`:
```python
from lola.agents.react import ReActAgent
from tools.custom_contract import CustomContractTool
from lola.utils.config import load_config
from web3 import Web3

class OnchainAgent(ReActAgent):
    def __init__(self):
        config = load_config()
        web3 = Web3(Web3.HTTPProvider(config["rpc_url"]))
        super().__init__(tools=[CustomContractTool(web3)], model=config["model"])
```

## Step 5: Run the Agent
```bash
poetry run lola run onchain_tool.agent.OnchainAgent "Get NFT floor price"
```

## Next Steps
- Explore `lola.chains` for more EVM abstractions.
- See [Onchain Analyst Example](../../examples/onchain_analyst/).
