# Quickstart

This guide helps you install LOLA OS and run your first agent.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/xai/lola-os
   cd lola-os
   ```
2. Install dependencies with Poetry:
   ```bash
   poetry install
   ```
3. Set PYTHONPATH:
   ```bash
   export PYTHONPATH=$(pwd)/python:$PYTHONPATH
   ```

## Run Your First Agent
1. Create a project:
   ```bash
   poetry run lola create my_project
   cd my_project
   poetry install
   ```
2. Configure `config.yaml`:
   ```yaml
   model: openai/gpt-4o
   api_key: your-api-key-here
   ```
3. Run the agent:
   ```bash
   poetry run lola run my_project.agent.BasicAgent "Test query"
   ```

## Next Steps
- Explore [Examples](../examples/) for advanced use cases.
- Read [Concepts](concepts/) to understand core components.
- Follow [Tutorials](tutorials/) to build custom agents.
