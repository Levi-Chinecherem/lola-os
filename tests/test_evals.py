# Standard imports
import pytest
import typing as tp

# Local
from lola.evals import AgentBenchmarker, GraphVisualizer, ScenarioRunner, AgenticSimulator, AdversarialTestingSuite
from lola.agents.react import ReActAgent
from lola.tools.human_input import HumanInputTool
from lola.core.graph import StateGraph
from lola.core.state import State

"""
File: Tests for evals module in LOLA OS TMVP 1 Phase 2.

Purpose: Verifies evaluation component initialization and functionality.
How: Uses pytest to test eval classes.
Why: Ensures robust testing, per Radical Reliability.
Full Path: lola-os/tests/test_evals.py
"""
def test_evals_functionality():
    """Test eval component functionality."""
    agent = ReActAgent(tools=[HumanInputTool()], model="openai/gpt-4o")
    graph = StateGraph(State())
    benchmarker = AgentBenchmarker()
    visualizer = GraphVisualizer()
    runner = ScenarioRunner()
    simulator = AgenticSimulator()
    adversarial = AdversarialTestingSuite()

    assert isinstance(benchmarker.run_benchmark(agent, ["test"]), dict)
    assert isinstance(visualizer.visualize(graph), dict)
    assert isinstance(runner.run_scenario(agent, "test"), dict)
    assert isinstance(simulator.simulate(agent, "test"), dict)
    assert isinstance(adversarial.test_adversarial(agent, ["test"]), dict)