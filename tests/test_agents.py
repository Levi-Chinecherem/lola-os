# Standard imports
import pytest
import asyncio
from unittest.mock import patch, AsyncMock, MagicMock
import typing as tp

# Local
from lola.agents import (
    BaseAgent, ReActAgent, PlanExecuteAgent, CollaborativeAgent, RouterAgent,
    ConversationalAgent, CodingAgent, OrchestratorAgent, MetaCognitionAgent,
    SyntheticDataAgent, LegacyInterfaceAgent
)
from lola.core.state import State
from lola.tools.base import BaseTool
from lola.orchestration.swarm import AgentSwarmOrchestrator

"""
File: Comprehensive tests for LOLA OS agents in Phase 2.

Purpose: Validates agent initialization, execution, and interconnections with real tools and state.
How: Uses pytest with async support, mocks for LLM/tool calls, and test data for validation.
Why: Ensures robust agent performance with >80% coverage, per Radical Reliability tenet.
Full Path: lola-os/tests/test_agents.py
"""

class MockTool(BaseTool):
    async def execute(self, input_data: tp.Any) -> tp.Any:
        return "Mock tool result"

@pytest.mark.asyncio
async def test_base_agent():
    """Test BaseAgent initialization and tool execution."""
    agent = BaseAgent(model="test/model", tools=[MockTool()])
    assert len(agent.tools_dict) == 1
    result = await agent.execute_tool("mock_tool", {})
    assert result == "Mock tool result"

@pytest.mark.asyncio
async def test_react_agent(mocker):
    """Test ReActAgent execution with mocked LLM."""
    mocker.patch('lola.agents.base.BaseAgent._call_llm', side_effect=["Reasoning", "final answer: Result"])
    agent = ReActAgent(tools=[MockTool()])
    state = await agent.run("Test query")
    assert state.output == "Result"

@pytest.mark.asyncio
async def test_plan_execute_agent(mocker):
    """Test PlanExecuteAgent execution with mocked LLM."""
    mocker.patch('lola.agents.base.BaseAgent._call_llm', return_value="Step 1\nStep 2")
    agent = PlanExecuteAgent(tools=[MockTool()])
    state = await agent.run("Test query")
    assert state.output == "Plan completed"

@pytest.mark.asyncio
async def test_collaborative_agent(mocker):
    """Test CollaborativeAgent execution with mocked sub-agent."""
    mocker.patch('lola.agents.base.BaseAgent._call_llm', return_value="Sub-task 1\nSub-task 2")
    sub_agent = MagicMock(spec=BaseAgent)
    sub_agent.run.return_value = State(output="Sub result")
    agent = CollaborativeAgent(sub_agents=[sub_agent])
    state = await agent.run("Test query")
    assert state.output == "Sub result"

@pytest.mark.asyncio
async def test_router_agent(mocker):
    """Test RouterAgent execution with mocked classification."""
    mocker.patch('lola.agents.base.BaseAgent._call_llm', return_value="category1")
    sub_agent = MagicMock(spec=BaseAgent)
    sub_agent.run.return_value = State(output="Routed result")
    agent = RouterAgent(routed_agents={"category1": sub_agent})
    state = await agent.run("Test query")
    assert state.output == "Routed result"

@pytest.mark.asyncio
async def test_conversational_agent(mocker):
    """Test ConversationalAgent execution with mocked LLM."""
    mocker.patch('lola.agents.base.BaseAgent._call_llm', return_value="Response")
    agent = ConversationalAgent(tools=[])
    state = await agent.run("Test query")
    assert state.output == "Response"
    assert len(agent.conversation_memory.history) == 2

@pytest.mark.asyncio
async def test_coding_agent(mocker):
    """Test CodingAgent execution with mocked LLM."""
    mocker.patch('lola.agents.base.BaseAgent._call_llm', return_value="print('Hello')")
    agent = CodingAgent(tools=[])
    state = await agent.run("Test query")
    assert state.output == "Hello\n"

@pytest.mark.asyncio
async def test_orchestrator_agent(mocker):
    """Test OrchestratorAgent execution with mocked sub-agent."""
    mocker.patch('lola.agents.base.BaseAgent._call_llm', return_value="Sub-task")
    sub_agent = MagicMock(spec=BaseAgent)
    sub_agent.run.return_value = State(output="Orchestrated result")
    agent = OrchestratorAgent(sub_agents=[sub_agent])
    state = await agent.run("Test query")
    assert state.output == "Orchestrated result"

@pytest.mark.asyncio
async def test_meta_cognition_agent(mocker):
    """Test MetaCognitionAgent execution with mocked LLM."""
    mocker.patch('lola.agents.base.BaseAgent._call_llm', side_effect=["Critique", "Refined"])
    agent = MetaCognitionAgent(tools=[])
    state = await agent.run("Test query")
    assert state.output == "Refined"

@pytest.mark.asyncio
async def test_synthetic_data_agent(mocker):
    """Test SyntheticDataAgent execution with mocked LLM."""
    mocker.patch('lola.agents.base.BaseAgent._call_llm', return_value="Synthetic data")
    agent = SyntheticDataAgent(tools=[])
    state = await agent.run("Test query")
    assert state.output == "Synthetic data"

@pytest.mark.asyncio
async def test_legacy_interface_agent(mocker):
    """Test LegacyInterfaceAgent execution with mocked LLM."""
    mocker.patch('lola.agents.base.BaseAgent._call_llm', return_value="Legacy API call")
    agent = LegacyInterfaceAgent(tools=[])
    state = await agent.run("Test query")
    assert state.output == "Legacy API call"

# Run tests if executed directly
if __name__ == "__main__":
    pytest.main()