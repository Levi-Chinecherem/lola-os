# Build Your First Agent

This tutorial guides you through creating a ReAct agent with LOLA OS.

## Step 1: Scaffold a Project

```bash
poetry run lola create my_agent --template react
cd my_agent
```

## Step 2: Configure the Agent

Edit `config.yaml`:
```yaml
openai_api_key: "your-openai-api-key"
```

## Step 3: Modify the Agent

Edit `agents/main.py` to add a custom tool:
```python
from lola.tools.web_search import WebSearchTool
agent = ReActAgent(tools=[WebSearchTool()], model=config.get("openai_api_key"))
```

## Step 4: Run the Agent

```bash
poetry run lola run agents/main.py "What is the weather today?"
```

## Expected Output

```
Agent output: The weather today in New York is sunny, 75°F.
```

## Next Steps

- Add more tools from `lola.tools`.
- Explore on-chain tools: [Building On-Chain Tools](building_onchain_tools.md).
