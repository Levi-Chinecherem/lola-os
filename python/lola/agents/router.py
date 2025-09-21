# Standard imports
import typing as tp

# Local
from .base import BaseTemplateAgent
from lola.core.state import State

"""
File: Defines the RouterAgent class for LOLA OS TMVP 1 Phase 2.

Purpose: Implements a router agent for directing queries to specialized agents.
How: Uses LLM to classify and route queries.
Why: Enables intelligent task delegation, per Choice by Design tenet.
Full Path: lola-os/python/lola/agents/router.py
Future Optimization: Migrate to Rust for fast routing (post-TMVP 1).
"""

class RouterAgent(BaseTemplateAgent):
    """RouterAgent: Routes queries to specialized agents. Does NOT persist state—use StateManager."""

    def __init__(self, routed_agents: tp.Dict[str, BaseTemplateAgent], model: str = "openai/gpt-4o"):
        """
        Initialize with routed agents and LLM model.

        Args:
            routed_agents: Dict of category to BaseTemplateAgent.
            model: LLM model string for litellm.
        """
        super().__init__(tools=[], model=model)
        self.routed_agents = routed_agents

    async def run(self, query: str) -> State:
        """
        Route the query to the appropriate agent.

        Args:
            query: User input string.
        Returns:
            State: Result from routed agent.
        Does Not: Handle fallback—use agnostic/fallback.py.
        """
        self.state.update({"query": query})
        # Inline: Classify query with LLM
        classification_prompt = f"Classify query into category: {query}\nCategories: {list(self.routed_agents.keys())}"
        category = await self._call_llm(classification_prompt)
        agent = self.routed_agents.get(category.strip(), None)
        if not agent:
            raise ValueError(f"No agent for category {category}")
        return await agent.run(query)

__all__ = ["RouterAgent"]