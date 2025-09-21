# Standard imports
import click
import asyncio
from pathlib import Path
import importlib.util
import sys

# Local imports
from lola.utils.logging import logger
from lola.agents.base import BaseAgent

"""
File: CLI command to run agents in development mode for LOLA OS TMVP 1 Phase 3.

Purpose: Executes a specified agent with a given query.
How: Uses click for arguments, asyncio for running agents.
Why: Enables developers to test agents locally, per Developer Sovereignty.
Full Path: lola-os/python/lola/cli/commands/run.py
"""

@click.command()
@click.argument('agent_path', type=click.Path(exists=True))
@click.argument('query')
def run(agent_path: str, query: str) -> None:
    """
    Run a LOLA OS agent with a query.

    Args:
        agent_path: Path to the agent Python file.
        query: Input query for the agent.

    Does not:
        Deploy the agent; only runs locally.
    """
    agent_file = Path(agent_path)
    if not agent_file.suffix == ".py":
        logger.error("Agent path must be a Python file.")
        raise click.Abort()

    # Dynamically import the agent module
    spec = importlib.util.spec_from_file_location("agent_module", agent_file)
    if not spec or not spec.loader:
        logger.error(f"Failed to load agent from {agent_file}.")
        raise click.Abort()

    module = importlib.util.module_from_spec(spec)
    sys.modules["agent_module"] = module
    spec.loader.exec_module(module)

    # Find the agent class
    agent_class = None
    for name in dir(module):
        obj = getattr(module, name)
        if isinstance(obj, type) and issubclass(obj, BaseAgent) and obj != BaseAgent:
            agent_class = obj
            break

    if not agent_class:
        logger.error(f"No valid BaseAgent subclass found in {agent_file}.")
        raise click.Abort()

    # Instantiate and run the agent
    agent = agent_class()
    result = asyncio.run(agent.run(query))
    logger.info(f"Agent output: {result.output}")