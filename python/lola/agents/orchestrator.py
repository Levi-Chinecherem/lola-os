# Standard imports
import typing as tp

# Local
from .base import BaseTemplateAgent
from lola.core.state import State
from lola.orchestration.swarm import AgentSwarmOrchestrator

"""
File: Defines the OrchestratorAgent class for LOLA OS TMVP 1 Phase 2.

Purpose: Implements an agent for managing sub-agents.
How: Uses AgentSwarmOrchestrator to delegate tasks.
Why: Enables hierarchical agent structures, per Choice by Design tenet.
Full Path: lola-os/python/lola/agents/orchestrator.py
Future Optimization: Migrate to Rust for scalable orchestration (post-TMVP 1).
"""

class OrchestratorAgent(BaseTemplateAgent):
    """OrchestratorAgent: Manages sub-agents for task delegation. Does NOT persist state—use StateManager."""

    def __init__(self, sub_agents: tp.List[BaseTemplateAgent], model: str = "openai/gpt-4o"):
        """
        Initialize with sub-agents and LLM model.

        Args:
            sub_agents: List of BaseTemplateAgent instances.
            model: LLM model string for litellm.
        """
        super().__init__(tools=[], model=model)
        self.orchestrator = AgentSwarmOrchestrator(sub_agents)

    async def run(self, query: str) -> State:
        """
        Delegate tasks to sub-agents.

        Args:
            query: User input string.
        Returns:
            State: Final state after delegation.
        Does Not: Handle blackboard—use orchestration/blackboard.py.
        """
        self.state.update({"query": query})
        # Inline: Decompose query with LLM
        decomposition = await self._call_llm(f"Decompose into sub-tasks: {query}")
        sub_tasks = decomposition.split("\n")
        # Inline: Delegate to orchestrator
        results = await self.orchestrator.run(sub_tasks)
        self.state.update({"output": results})
        return self.state

__all__ = ["OrchestratorAgent"]