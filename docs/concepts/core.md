# Core Concepts

This section explains the core components of LOLA OS.

## BaseAgent

The `BaseAgent` class (`lola.agents.base`) is the foundation for all agents. It integrates with `litellm` for LLM calls and supports tool execution.

## StateGraph

The `StateGraph` (`lola.core.graph`) orchestrates agent workflows using a directed acyclic graph (DAG). Nodes represent tasks, and edges define transitions.

## Tools

Tools (`lola.tools`) like `WebSearchTool` and `ContractCallTool` enable agents to interact with external systems (web, EVM).

## RAG

Retrieval-Augmented Generation (`lola.rag`) uses Pinecone and LlamaIndex for enhanced query processing.

## Next Steps

- Build a ReAct agent: [Build Your First Agent](../tutorials/build_your_first_agent.md)
- Create on-chain tools: [Building On-Chain Tools](../tutorials/building_onchain_tools.md)
