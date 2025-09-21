# Quickstart

This guide helps you install LOLA OS and run your first agent.

## Prerequisites

- Python 3.10+
- Poetry
- API keys for OpenAI and Pinecone
- Ethereum testnet RPC (e.g., Sepolia)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-org/lola-os.git
   cd lola-os
   ```

2. Install dependencies:
   ```bash
   poetry install
   ```

3. Set environment variables:
   ```bash
   export PINECONE_API_KEY="your-key"
   export OPENAI_API_KEY="your-key"
   export WEB3_PROVIDER_URI="https://sepolia.infura.io/v3/your-infura-key"
   export REDIS_URL="redis://localhost:6379/0"
   ```

## Run an Example

Try the research agent:
```bash
cd examples/research_agent
poetry run python agent.py "What is the capital of France?"
```

Expected output:
```
Research result: The capital of France is Paris.
```

## Next Steps

- Explore [Core Concepts](concepts/core.md)
- Follow [Tutorials](tutorials/build_your_first_agent.md)
