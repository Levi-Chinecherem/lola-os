# Standard imports
import asyncio
from typing import List

# Local imports
from lola.agents.react import ReActAgent
from lola.tools.web_search import WebSearchTool
from lola.rag.multimodal import MultiModalRetriever
from lola.utils.config import config, load_config
from lola.utils.logging import logger
from lola.core.state import State

"""
File: Research agent example for LOLA OS TMVP 1 Phase 4.

Purpose: Demonstrates a ReAct agent performing web searches and RAG queries.
How: Uses ReActAgent with WebSearchTool and MultiModalRetriever for real queries.
Why: Showcases developer-friendly agent creation, per Developer Sovereignty.
Full Path: lola-os/examples/research_agent/agent.py
"""

async def main(query: str) -> str:
    """
    Run a research agent to answer a query using web search and RAG.

    Args:
        query: Input query for research.

    Returns:
        Final answer as a string.

    Does not:
        Modify external systems; only performs read operations.
    """
    # Load configuration
    config_path = "config.yaml"
    load_config(config_path)

    # Initialize tools
    web_tool = WebSearchTool()
    retriever = MultiModalRetriever(pinecone_api_key=config.get("pinecone_api_key"))
    
    # Initialize agent with tools
    agent = ReActAgent(
        model=config.get("openai_api_key"),
        tools=[web_tool, retriever]
    )
    
    # Run agent
    logger.info(f"Processing query: {query}")
    state = await agent.run(query)
    
    return state.output

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        logger.error("Please provide a query as an argument.")
        sys.exit(1)
    
    result = asyncio.run(main(sys.argv[1]))
    logger.info(f"Research result: {result}")