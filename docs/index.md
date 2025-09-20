# LOLA OS TMVP 1 Documentation

LOLA OS (Layered Orchestration for Logic and Automation, or Live Onchain Logical Agents) is an open-source framework for building read-only AI agents with EVM-native capabilities. TMVP 1 delivers a Python SDK (`pip install lola-os`) with modular components, agent templates, and CLI tools.

## Overview
- **Purpose**: Enable developers to create AI agents for on-chain and off-chain tasks.
- **Features**: ReAct agents, EVM read tools, advanced RAG, multi-agent orchestration, guardrails, and more.
- **Philosophy**: Developer Sovereignty, EVM-Native, Radical Reliability, Choice by Design.

## Getting Started
See [Quickstart](quickstart.md) for installation and your first agent.

## Key Components
- **Core**: Base abstractions (`agent.py`, `graph.py`, `state.py`, `memory.py`).
- **Agents**: Templates like ReAct, PlanExecute, and Orchestrator.
- **Tools**: Web search, EVM reads, human input, and more.
- **Chains**: EVM read abstractions via `web3.py`.
- **Docs**: Explore [Concepts](concepts/) and [Tutorials](tutorials/).

## Examples
Try the [research agent](../examples/research_agent/) or [onchain analyst](../examples/onchain_analyst/).

## Contributing
See [Contributing Guide](contributing.md) in the project root.
