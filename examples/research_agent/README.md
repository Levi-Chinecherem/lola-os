# Research Agent Example

This example demonstrates a ReAct-based agent for web research using LOLA OS.

## Setup
1. Install LOLA OS:
   ```bash
   poetry install
   ```
2. Set API key in `config.yaml`:
   ```yaml
   model: openai/gpt-4o
   api_key: your-api-key-here
   ```
3. Run the agent:
   ```bash
   poetry run python agent.py "Research the latest AI trends"
   ```

## Features
- Uses `WebSearchTool` for querying the web.
- Integrates `HumanInputTool` for HITL interactions.
- Configurable via `config.yaml` with `litellm` model support.

## Output
The agent returns a `State` object with research results, logged to stdout.
