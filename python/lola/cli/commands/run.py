# Standard imports
import click
import asyncio
from importlib import import_module
import sys

# Local
from lola.utils.logging import logger
from lola.utils.config import load_config
from lola.agents.react import ReActAgent
from lola.tools.human_input import HumanInputTool
from lola.core.state import State

"""
File: CLI command to run a LOLA OS agent in development mode.

Purpose: Executes a specified agent with a given query.
How: Loads agent and config dynamically; runs async execution.
Why: Enables rapid iteration for developers, per Developer Sovereignty.
Full Path: lola-os/python/lola/cli/commands/run.py
"""

@click.command()
@click.argument("agent_path")
@click.argument("query")
def run(agent_path: str, query: str) -> None:
    """
    Run a LOLA OS agent with a query.

    Args:
        agent_path: Path to agent module (e.g., my_agent.agent).
        query: Input query for the agent.
    Does Not: Persist state—use StateManager for persistence.
    """
    config = load_config()
    model = config.get("model", "openai/gpt-4o")

    try:
        module_name, agent_class = agent_path.rsplit(".", 1)
        logger.debug(f"Attempting to import module: {module_name}, class: {agent_class}")
        logger.debug(f"Current sys.path: {sys.path}")
        module = import_module(module_name)
        agent = getattr(module, agent_class)(tools=[HumanInputTool()], model=model)
    except Exception as e:
        logger.error(f"Failed to load agent {agent_path}: {str(e)}")
        raise click.Abort()

    async def execute_agent():
        try:
            result = await agent.run(query)
            logger.info(f"Agent result: {result.data}")
            return result
        except Exception as e:
            logger.error(f"Agent execution failed: {str(e)}")
            raise click.Abort()

    asyncio.run(execute_agent())