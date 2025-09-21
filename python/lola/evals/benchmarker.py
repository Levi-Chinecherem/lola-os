# Standard imports
import typing as tp
import asyncio
import statistics

# Local
from lola.agents.base import BaseTemplateAgent
from lola.core.state import State
from lola.utils.logging import logger

"""
File: Defines the AgentBenchmarker class for LOLA OS TMVP 1 Phase 2.

Purpose: Benchmarks agent performance with real metrics like accuracy and latency.
How: Runs multiple test cases asynchronously, calculates statistics.
Why: Ensures reliable agent evaluation, per Radical Reliability tenet.
Full Path: lola-os/python/lola/evals/benchmarker.py
Future Optimization: Migrate to Rust for high-throughput benchmarking (post-TMVP 1).
"""

class AgentBenchmarker:
    """AgentBenchmarker: Benchmarks agents with test cases. Does NOT persist results—use StateManager."""

    async def run_benchmark(self, agent: BaseTemplateAgent, test_cases: tp.List[tp.Dict[str, str]]) -> dict:
        """
        Runs benchmark on an agent.

        Args:
            agent: BaseTemplateAgent instance.
            test_cases: List of dicts with 'query' and 'expected' output.

        Returns:
            Dict with metrics (accuracy, average_latency).

        Does Not: Visualize—use visualizer.py.
        """
        accuracies = []
        latencies = []
        for case in test_cases:
            start_time = time.time()
            state = await agent.run(case['query'])
            end_time = time.time()
            accuracy = 1 if state.output == case['expected'] else 0
            accuracies.append(accuracy)
            latencies.append(end_time - start_time)
        return {
            "accuracy": statistics.mean(accuracies),
            "average_latency": statistics.mean(latencies)
        }

__all__ = ["AgentBenchmarker"]