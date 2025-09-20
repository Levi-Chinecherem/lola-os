# Tutorial: Build Your First Agent

This tutorial guides you through building a ReAct-based agent with LOLA OS.

## Step 1: Create a Project
```bash
poetry run lola create my_agent
cd my_agent
poetry install
```

## Step 2: Configure the Agent
Edit `config.yaml`:
```yaml
model: openai/gpt-4o
api_key: your-api-key-here
```

## Step 3: Customize the Agent
Modify `agent.py`:
```python
from lola.agents.react import ReActAgent
from lola.tools.web_search import WebSearchTool
from lola.utils.logging import logger

class MyAgent(ReActAgent):
    def __init__(self):
        super().__init__(tools=[WebSearchTool()], model="openai/gpt-4o")
        logger.info("Initialized MyAgent")
```

## Step 4: Run the Agent
```bash
poetry run lola run my_agent.agent.MyAgent "Research AI trends"
```

## Next Steps
- Add more tools from `lola.tools`.
- Explore [Concepts](../concepts/) for advanced features.
