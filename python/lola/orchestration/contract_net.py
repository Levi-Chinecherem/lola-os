# Standard imports
import typing as tp
import asyncio

# Local
from lola.agents.base import BaseTemplateAgent
from lola.core.state import State
from lola.utils.logging import logger

"""
File: Defines the ContractNetProtocol class for LOLA OS TMVP 1 Phase 2.

Purpose: Implements contract net protocol for agent task allocation.
How: Uses LLM to bid on tasks, allocates to best bid.
Why: Enables decentralized task assignment, per Choice by Design tenet.
Full Path: lola-os/python/lola/orchestration/contract_net.py
Future Optimization: Migrate to Rust for secure bidding (post-TMVP 1).
"""

class ContractNetProtocol:
    """ContractNetProtocol: Manages task bidding among agents. Does NOT persist bids—use StateManager."""

    def __init__(self, agents: tp.List[BaseTemplateAgent]):
        """
        Initialize with agents.

        Args:
            agents: List of BaseTemplateAgent instances.
        """
        self.agents = agents
        logger.info(f"Initialized ContractNetProtocol with {len(agents)} agents.")

    async def allocate_task(self, task: str) -> State:
        """
        Allocates task via bidding.

        Args:
            task: Task description.

        Returns:
            State from winning agent.

        Does Not: Handle ties—random selection.
        """
        bids = []
        for agent in self.agents:
            bid_prompt = f"Bid on task: {task} (0-10 score)"
            bid = await agent._call_llm(bid_prompt)
            bids.append((int(bid), agent))
        winner = max(bids, key=lambda x: x[0])[1]
        return await winner.run(task)

__all__ = ["ContractNetProtocol"]