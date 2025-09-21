# Standard imports
import typing as tp
import asyncio

# Local
from lola.agents.base import BaseTemplateAgent
from lola.core.state import State
from lola.utils.logging import logger

"""
File: Defines the GroupChatManager class for LOLA OS TMVP 1 Phase 2.

Purpose: Manages group chat among agents.
How: Moderates turns with LLM, collects responses.
Why: Enables productive multi-agent discussions, per Choice by Design tenet.
Full Path: lola-os/python/lola/orchestration/group_chat.py
Future Optimization: Migrate to Rust for real-time chat (post-TMVP 1).
"""

class GroupChatManager:
    """GroupChatManager: Manages group chat for agents. Does NOT persist chat—use StateManager."""

    def __init__(self, agents: tp.List[BaseTemplateAgent], moderator_model: str = "openai/gpt-4o"):
        """
        Initialize with agents and moderator model.

        Args:
            agents: List of BaseTemplateAgent instances.
            moderator_model: LLM model for moderation.
        """
        self.agents = agents
        self.moderator_model = moderator_model
        logger.info(f"Initialized GroupChatManager with {len(agents)} agents.")

    async def moderate(self, topic: str, rounds: int = 3) -> State:
        """
        Moderates a group chat.

        Args:
            topic: Discussion topic.
            rounds: Number of discussion rounds.

        Returns:
            State with chat summary.

        Does Not: Handle conflicts—use contract_net.py.
        """
        state = State(query=topic, history=[])
        for round_num in range(rounds):
            for agent in self.agents:
                prompt = f"Discuss round {round_num}: {topic}\nHistory: {state.history}"
                response = await agent._call_llm(prompt, model=self.moderator_model)
                state.history.append({"agent": agent.__class__.__name__, "response": response})
        summary = await agent._call_llm(f"Summarize discussion: {state.history}", model=self.moderator_model)
        state.update(output=summary)
        return state

__all__ = ["GroupChatManager"]