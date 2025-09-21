# Standard imports
import asyncio

# Local imports
from lola.agents.react import ReActAgent
from lola.tools.web_search import WebSearchTool
from lola.utils.config import config
from lola.utils.logging import logger

"""
File: Template React agent for LOLA OS TMVP 1 Phase 3.

Purpose: Provides a default ReAct agent for scaffolded projects.
How: Implements a simple ReAct agent with a web search tool.
Why: Demonstrates agent functionality, per Developer Sovereignty.
Full Path: lola-os/python/lola/cli/templates/react/agents/main.py
"""

async def main(query: str) -> str:
    """
    Run a ReAct agent with a web search tool.

    Args:
        query: Input query for the agent.

    Returns:
        Agent output as a string.
    """
    agent = ReActAgent(tools=[WebSearchTool()], model=config.get("openai_api_key"))
    result = await agent.run(query)
    return result.output

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        logger.error("Please provide a query as an argument.")
        sys.exit(1)
    result = asyncio.run(main(sys.argv[1]))
    logger.info(f"Agent output: {result}")