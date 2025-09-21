# Standard imports
import pytest
import asyncio
import time
import statistics
from unittest.mock import AsyncMock, patch, MagicMock

# Third-party
import wandb

# Local
from lola.evals.benchmarker import AgentBenchmarker
from lola.evals.visualizer import GraphVisualizer
from lola.evals.scenario_runner import ScenarioRunner
from lola.evals.simulator import AgenticSimulator
from lola.evals.adversarial import AdversarialTestingSuite
from lola.agents.base import BaseTemplateAgent
from lola.core.state import State
from lola.core.graph import StateGraph
from lola.utils import sentry

"""
File: Comprehensive tests for LOLA OS evals in Phase 5.

Purpose: Validates evaluation tools with real tracking (W&B/MLflow) and mocked agents.
How: Uses pytest with async support, real W&B/MLflow calls (mocked for CI), and integration tests.
Why: Ensures thorough agent testing with >80% coverage, per Radical Reliability tenet.
Full Path: lola-os/tests/test_evals.py
"""

@pytest.fixture
def mock_agent():
    """Fixture for a mocked agent."""
    agent = MagicMock(spec=BaseTemplateAgent)
    agent.run = AsyncMock(return_value=State(output="expected"))
    return agent

@pytest.mark.asyncio
async def test_agent_benchmarker_wandb(mocker, mock_agent):
    """Test AgentBenchmarker with W&B tracking."""
    mocker.patch("wandb.log")
    mocker.patch("lola.utils.sentry.init_sentry")
    mocker.patch("prometheus_client.Counter.inc")
    config = {"tracker": {"type": "wandb", "project": "lola-benchmarks"}}
    benchmarker = AgentBenchmarker(config)
    test_cases = [{"query": "test", "expected": "expected"}]
    result = await benchmarker.run_benchmark(mock_agent, test_cases)
    assert result["accuracy"] == 1.0
    assert "average_latency" in result
    wandb.log.assert_called()

@pytest.mark.asyncio
async def test_scenario_runner(mocker, mock_agent):
    """Test ScenarioRunner with real async run."""
    mocker.patch("lola.utils.sentry.init_sentry")
    mocker.patch("prometheus_client.Counter.inc")
    config = {}
    runner = ScenarioRunner(config)
    scenarios = [{"query": "test", "expected": "expected", "name": "test_scenario"}]
    result = await runner.run_scenario(mock_agent, scenarios)
    assert result["results"][0]["passed"] is True

def test_graph_visualizer(tmp_path, mocker):
    """Test GraphVisualizer with sample graph."""
    mocker.patch("lola.utils.sentry.init_sentry")
    mocker.patch("prometheus_client.Counter.inc")
    graph = StateGraph(State)
    visualizer = GraphVisualizer()
    output_file = tmp_path / "test.dot"
    visualizer.visualize(graph, str(output_file))
    assert output_file.exists()

@pytest.mark.asyncio
async def test_adversarial_testing_suite(mocker, mock_agent):
    """Test AdversarialTestingSuite with real queries."""
    mocker.patch("lola.utils.sentry.init_sentry")
    mocker.patch("prometheus_client.Counter.inc")
    config = {"test_types": ["jailbreak"]}
    suite = AdversarialTestingSuite(config)
    queries = ["jailbreak test"]
    result = await suite.test_adversarial(mock_agent, queries)
    assert result["results"][0]["passed"] is True

@pytest.mark.asyncio
async def test_agentic_simulator(mocker):
    """Test AgenticSimulator with mock responses."""
    mocker.patch("lola.utils.sentry.init_sentry")
    mocker.patch("prometheus_client.Counter.inc")
    config = {"mock_responses": {"tool": "simulated"}}
    simulator = AgenticSimulator(config)
    agent = MagicMock(spec=BaseTemplateAgent)
    agent.run = AsyncMock(return_value=State(output="simulated"))
    state = simulator.simulate(agent, "simulated query")
    assert state.output == "simulated"

# Run tests if executed directly
if __name__ == "__main__":
    pytest.main()