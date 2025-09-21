# Standard imports
import typing as tp
import asyncio

# Local
from lola.agents.base import BaseTemplateAgent
from lola.core.state import State

"""
File: Defines the ScenarioRunner class for LOLA OS TMVP 1 Phase 2.

Purpose: Runs agents through predefined scenarios for testing.
How: Executes sequences of queries and validates outputs.
Why: Ensures agent robustness in real scenarios, per Radical Reliability tenet.
Full Path: lola-os/python/lola/evals/scenario_runner.py
Future Optimization: Migrate to Rust for high-volume scenario running (post-TMVP 1).
"""

class ScenarioRunner:
    """ScenarioRunner: Runs agent scenarios. Does NOT persist results—use StateManager."""

    async def run_scenario(self, agent: BaseTemplateAgent, scenarios: tp.List[tp.Dict[str, tp.Any]]) -> dict:
        """
        Runs a scenario on an agent.

        Args:
            agent: BaseTemplateAgent instance.
            scenarios: List of dicts with 'query' and 'expected'.

        Returns:
            Dict with pass/fail results.

        Does Not: Simulate failures—use simulator.py.
        """
        results = []
        for scenario in scenarios:
            state = await agent.run(scenario['query'])
            passed = state.output == scenario['expected']
            results.append({"passed": passed, "actual": state.output})
        return {"results": results}

__all__ = ["ScenarioRunner"]