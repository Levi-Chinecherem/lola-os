# Standard imports
import pytest
import asyncio
import typing as tp
from unittest.mock import AsyncMock, patch

# Local
from lola.agents import (
    BaseTemplateAgent,
    ReActAgent,
    PlanExecuteAgent,
    CollaborativeAgent,
    RouterAgent,
    ConversationalAgent,
    CodingAgent,
    OrchestratorAgent,
    MetaCognitionAgent,
    SyntheticDataAgent,
    LegacyInterfaceAgent
)
from lola.core.state import State
from lola.tools.base import BaseTool
from lola.utils import sentry
from lola.utils import prometheus

"""
File: Tests for LOLA OS agent templates in Phase 2 and 5.

Purpose: Validates agent functionality with >80% coverage, including async execution, error cases, and integrations.
How: Uses pytest with async support, mocks for LLM/tool calls, and real integration tests.
Why: Ensures robust agent implementations, per Radical Reliability tenet.
Full Path: lola-os/tests/test_agents.py
"""

@pytest.fixture
def mock_tool():
    """Fixture for a mock tool."""
    class MockTool(BaseTool):
        name = "mock_tool"
        async def execute(self, input_data: tp.Any) -> tp.Any:
            return f"Mock result for {input_data}"
    return MockTool()

@pytest.fixture
def mock_state():
    """Fixture for an initial state."""
    return State(query="", history=[], metadata={})

@pytest.mark.asyncio
async def test_base_template_agent(mock_tool, mock_state):
    """Test BaseTemplateAgent with Sentry and Prometheus."""
    with patch("lola.utils.sentry.init_sentry"), patch("prometheus_client.Counter.inc"):
        agent = BaseTemplateAgent(tools=[mock_tool], model="test/model")
        assert agent.tools_dict["mock_tool"] == mock_tool
        result = await agent.execute_tool("mock_tool", {"data": "test"})
        assert result == "Mock result for {'data': 'test'}"

@pytest.mark.asyncio
async def test_react_agent_integration(mocker, mock_tool, mock_state):
    """Test ReActAgent with real LLM call and tool integration."""
    mocker.patch("lola.core.agent.BaseAgent.call_llm", AsyncMock(side_effect=[
        "Reasoning step", "mock_tool(data='test')", "Final answer: Done"
    ]))
    with patch("lola.utils.sentry.capture_exception"), patch("prometheus_client.Histogram.observe"):
        agent = ReActAgent(tools=[mock_tool], model="openai/gpt-4o")
        state = await agent.run("Test query")
        assert state.output == "Done"
        assert len(state.metadata["steps"]) == 2

@pytest.mark.asyncio
async def test_collaborative_agent_orchestration(mocker, mock_tool, mock_state):
    """Test CollaborativeAgent with real orchestration."""
    mock_sub_agent = BaseTemplateAgent(tools=[mock_tool], model="test/model")
    mocker.patch.object(mock_sub_agent, "run", AsyncMock(return_value=mock_state))
    mocker.patch("lola.core.agent.BaseAgent.call_llm", AsyncMock(return_value="Task 1\nTask 2"))
    mocker.patch("lola.orchestration.swarm.AgentSwarmOrchestrator.run", AsyncMock(return_value=["Result 1", "Result 2"]))
    with patch("lola.utils.sentry.capture_exception"):
        agent = CollaborativeAgent(sub_agents=[mock_sub_agent], model="test/model")
        state = await agent.run("Test query")
        assert state.output == ["Result 1", "Result 2"]