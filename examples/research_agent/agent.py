# Standard imports
from typing import List
import asyncio

# Third-party imports
from pydantic import BaseModel

# Local imports
from lola.agents.react import ReActAgent
from lola.tools.web_search import WebSearchTool
from lola.tools.human_input import HumanInputTool
from lola.utils.logging import logger
from lola.utils.config import load_config

"""
File: Research agent example for LOLA OS TMVP 1 Phase 4.

Purpose: Demonstrates a ReAct-based agent for web research tasks.
How: Uses WebSearchTool and HumanInputTool with ReActAgent.
Why: Showcases agent capabilities for developer onboarding, per Developer Sovereignty.
Full Path: lola-os/examples/research_agent/agent.py
"""

class ResearchAgent(ReActAgent):
    """A ReAct-based agent for web research tasks."""

    def __init__(self, tools: List = None, model: str = "openai/gpt-4o"):
        """
        Initialize the research agent.

        Args:
            tools: List of tools (defaults to WebSearchTool, HumanInputTool).
            model: LLM model name (defaults to openai/gpt-4o).
        """
        config = load_config("examples/research_agent/config.yaml")
        model = config.get("model", model)
        tools = tools or [WebSearchTool(), HumanInputTool()]
        super().__init__(tools=tools, model=model)
        logger.info(f"Initialized ResearchAgent with model {model}")

async def main(query: str):
    """Run the research agent with a query."""
    agent = ResearchAgent()
    result = await agent.run(query)
    logger.info(f"Research result: {result.data}")
    return result

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python agent.py <query>")
        sys.exit(1)
    asyncio.run(main(sys.argv[1]))