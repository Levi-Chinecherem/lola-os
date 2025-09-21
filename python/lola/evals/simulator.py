# Standard imports
import typing as tp
from unittest.mock import Mock

# Local
from lola.agents.base import BaseTemplateAgent
from lola.core.state import State

"""
File: Defines the AgenticSimulator class for LOLA OS TMVP 1 Phase 2.

Purpose: Simulates agent interactions with mock environments.
How: Uses unittest.mock to simulate API/tools.
Why: Tests agents in controlled settings, per Radical Reliability tenet.
Full Path: lola-os/python/lola/evals/simulator.py
Future Optimization: Migrate to Rust for complex simulations (post-TMVP 1).
"""

class AgenticSimulator:
    """AgenticSimulator: Simulates agent environments. Does NOT persist simulations—use StateManager."""

    def simulate(self, agent: BaseTemplateAgent, mock_responses: tp.Dict[str, tp.Any]) -> State:
        """
        Simulates agent run with mock responses.

        Args:
            agent: BaseTemplateAgent instance.
            mock_responses: Dict of tool name to mock response.

        Returns:
            Simulated state.

        Does Not: Handle adversarial—use adversarial.py.
        """
        # Inline: Mock tool execution
        for tool in agent.tools:
            tool.execute = Mock(return_value=mock_responses.get(tool.name, "mock_result"))
        return agent.run("simulated query")

__all__ = ["AgenticSimulator"]