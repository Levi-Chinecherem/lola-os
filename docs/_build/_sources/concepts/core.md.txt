# Core Abstractions

The core module (`python/lola/core/`) provides foundational abstractions for LOLA OS.

## Components
- **BaseAgent (`agent.py`)**: Abstract base class for all agents, defining the `run` interface.
- **StateGraph (`graph.py`)**: Manages agent workflows with nodes and edges.
- **State (`state.py`)**: Pydantic model for agent state, ensuring type safety.
- **StateManager (`memory.py`)**: Persists state across sessions.

## Usage
```python
from lola.core.agent import BaseAgent
from lola.core.state import State

class CustomAgent(BaseAgent):
    async def run(self, query: str) -> State:
        return State(data={"result": f"Processed: {query}"})
```

## Why?
These abstractions enable modular, extensible agent development, per Choice by Design.
