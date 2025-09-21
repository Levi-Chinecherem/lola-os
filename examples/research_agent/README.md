# LOLA OS Research Agent Example

This example demonstrates a ReAct-based research agent that uses web search and RAG (Retrieval-Augmented Generation) to answer queries.

## Prerequisites

- Python 3.10+
- Poetry installed
- Pinecone and OpenAI API keys

## Setup

1. Install dependencies:
   ```bash
   poetry install
   ```

2. Set environment variables or update `config.yaml`:
   ```bash
   export PINECONE_API_KEY="your-key"
   export OPENAI_API_KEY="your-key"
   ```

3. Run the agent:
   ```bash
   poetry run python agent.py "What is the capital of France?"
   ```

## Expected Output

The agent will perform a web search and RAG query, returning an answer like:
```
Research result: The capital of France is Paris.
```

## Next Steps

- Modify `agent.py` to add more tools or customize the ReAct logic.
- Explore the LOLA OS documentation for advanced features.
