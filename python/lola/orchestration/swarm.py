# Standard imports
import typing as tp
import asyncio

# Local
from lola.agents.base import BaseTemplateAgent
from lola.core.state import State
from lola.utils.logging import logger

"""
File: Defines the AgentSwarmOrchestrator class for LOLA OS TMVP 1 Phase 2.

Purpose: Orchestrates a swarm of agents for distributed tasks.
How: Uses asyncio to run agents concurrently, merges results into shared state.
Why: Enables scalable multi-agent systems, per Choice by Design tenet.
Full Path: lola-os/python/lola/orchestration/swarm.py
Future Optimization: Migrate to Rust for true parallelism (post-TMVP 1).
"""

class AgentSwarmOrchestrator:
    """AgentSwarmOrchestrator: Manages swarm of agents for collaborative execution. Does NOT persist state—use StateManager."""

    def __init__(self, agents: tp.List[BaseTemplateAgent]):
        """
        Initialize with a list of agents.

        Args:
            agents: List of BaseTemplateAgent instances.

        Does Not: Validate agents—caller must ensure compatibility.
        """
        self.agents = agents
        logger.info(f"Initialized AgentSwarmOrchestrator with {len(agents)} agents.")

    async def run(self, query: str) -> State:
        """
        Runs the swarm on a query, merging results.

        Args:
            query: Input query for all agents.

        Returns:
            Merged State from all agents.

        Does Not: Handle agent failures—use guardrails/prompt_shield.py.
        """
        tasks = [agent.run(query) for agent in self.agents]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        merged_state = State()
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Agent failed: {result}")
            else:
                merged_state.history.extend(result.history)
                merged_state.output = result.output  # Last agent's output overrides; improve in TMVP 2
        return merged_state

__all__ = ["AgentSwarmOrchestrator"]