# Standard imports
import pytest
import asyncio
from unittest.mock import MagicMock

# Local
from lola.evals.benchmarker import AgentBenchmarker
from lola.evals.visualizer import GraphVisualizer
from lola.evals.scenario_runner import ScenarioRunner
from lola.evals.simulator import AgenticSimulator
from lola.evals.adversarial import AdversarialTestingSuite
from lola.agents.base import BaseTemplateAgent
from lola.core.state import State
from lola.core.graph import StateGraph

"""
File: Comprehensive tests for LOLA OS evals in Phase 2.

Purpose: Validates evaluation tools with real metrics and mocked agents.
How: Uses pytest with async support, MagicMock for agents.
Why: Ensures thorough agent testing with >80% coverage, per Radical Reliability tenet.
Full Path: lola-os/tests/test_evals.py
"""

@pytest.mark.asyncio
async def test_agent_benchmarker():
    """Test AgentBenchmarker with mocked agent."""
    agent = MagicMock(spec=BaseTemplateAgent)
    agent.run.return_value = State(output="expected")
    benchmarker = AgentBenchmarker()
    test_cases = [{"query": "test", "expected": "expected"}]
    result = await benchmarker.run_benchmark(agent, test_cases)
    assert result["accuracy"] == 1.0
    assert "average_latency" in result

def test_graph_visualizer(tmp_path):
    """Test GraphVisualizer with sample graph."""
    graph = StateGraph(State)
    visualizer = GraphVisualizer()
    visualizer.visualize(graph, str(tmp_path / "test.dot"))
    assert (tmp_path / "test.dot").exists()

@pytest.mark.asyncio
async def test_scenario_runner():
    """Test ScenarioRunner with mocked agent."""
    agent = MagicMock(spec=BaseTemplateAgent)
    agent.run.return_value = State(output="expected")
    runner = ScenarioRunner()
    scenarios = [{"query": "test", "expected": "expected"}]
    result = await runner.run_scenario(agent, scenarios)
    assert result["results"][0]["passed"] is True

def test_agentic_simulator():
    """Test AgenticSimulator with mocked agent."""
    agent = MagicMock(spec=BaseTemplateAgent)
    simulator = AgenticSimulator()
    mock_responses = {"tool": "simulated"}
    state = simulator.simulate(agent, mock_responses)
    assert state.output == "simulated"

@pytest.mark.asyncio
async def test_adversarial_testing_suite():
    """Test AdversarialTestingSuite with mocked agent."""
    agent = MagicMock(spec=BaseTemplateAgent)
    agent.run.return_value = State(output="safe")
    suite = AdversarialTestingSuite()
    queries = ["jailbreak test"]
    result = await suite.test_adversarial(agent, queries)
    assert result["results"][0]["passed"] is True

# Run tests if executed directly
if __name__ == "__main__":
    pytest.main()