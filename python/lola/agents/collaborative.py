# Standard imports
import typing as tp

# Local
from .base import BaseTemplateAgent
from lola.core.state import State
from lola.orchestration.swarm import AgentSwarmOrchestrator

"""
File: Defines the CollaborativeAgent class for LOLA OS TMVP 1 Phase 2.

Purpose: Implements a collaborative agent pattern for multi-agent teamwork.
How: Uses AgentSwarmOrchestrator to coordinate sub-agents for task decomposition.
Why: Enables team-based problem-solving, per Choice by Design tenet.
Full Path: lola-os/python/lola/agents/collaborative.py
Future Optimization: Migrate to Rust for high-throughput collaboration (post-TMVP 1).
"""

class CollaborativeAgent(BaseTemplateAgent):
    """CollaborativeAgent: Implements multi-agent collaboration. Does NOT persist state—use StateManager."""

    def __init__(self, sub_agents: tp.List[BaseTemplateAgent], model: str = "openai/gpt-4o"):
        """
        Initialize with sub-agents and LLM model.

        Args:
            sub_agents: List of BaseTemplateAgent instances for collaboration.
            model: LLM model string for litellm.
        """
        super().__init__(tools=[], model=model)
        self.orchestrator = AgentSwarmOrchestrator(sub_agents)

    async def run(self, query: str) -> State:
        """
        Execute collaborative task.

        Args:
            query: User input string.
        Returns:
            State: Final state after collaboration.
        Does Not: Handle contract negotiation—use orchestration/contract_net.py.
        """
        self.state.update({"query": query})
        # Inline: Decompose task with LLM
        decomposition = await self._call_llm(f"Decompose into sub-tasks: {query}")
        sub_tasks = decomposition.split("\n")
        # Inline: Assign sub-tasks to sub-agents
        results = await self.orchestrator.run(sub_tasks)
        self.state.update({"output": results})
        return self.state

__all__ = ["CollaborativeAgent"]