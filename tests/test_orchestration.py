# Standard imports
import pytest
import asyncio
from unittest.mock import MagicMock

# Local
from lola.orchestration.swarm import AgentSwarmOrchestrator
from lola.orchestration.contract_net import ContractNetProtocol
from lola.orchestration.blackboard import BlackboardSystem
from lola.orchestration.group_chat import GroupChatManager
from lola.agents.base import BaseTemplateAgent
from lola.core.state import State

"""
File: Comprehensive tests for LOLA OS orchestration in Phase 2.

Purpose: Validates multi-agent coordination with real async and mocked agents.
How: Uses pytest with async support, MagicMock for agents.
Why: Ensures robust multi-agent systems with >80% coverage, per Choice by Design tenet.
Full Path: lola-os/tests/test_orchestration.py
"""

@pytest.mark.asyncio
async def test_agent_swarm_orchestrator():
    """Test AgentSwarmOrchestrator with mocked agents."""
    agent1 = MagicMock(spec=BaseTemplateAgent)
    agent2 = MagicMock(spec=BaseTemplateAgent)
    agent1.run.return_value = State(output="result1")
    agent2.run.return_value = State(output="result2")
    orchestrator = AgentSwarmOrchestrator([agent1, agent2])
    state = await orchestrator.run("Test query")
    assert state.output == "result2"  # Last result; improve merging in TMVP 2

@pytest.mark.asyncio
async def test_contract_net_protocol():
    """Test ContractNetProtocol with mocked agents."""
    agent1 = MagicMock(spec=BaseTemplateAgent)
    agent2 = MagicMock(spec=BaseTemplateAgent)
    agent1._call_llm.return_value = "5"
    agent2._call_llm.return_value = "8"
    agent2.run.return_value = State(output="winner result")
    protocol = ContractNetProtocol([agent1, agent2])
    state = await protocol.allocate_task("Test task")
    assert state.output == "winner result"

def test_blackboard_system():
    """Test BlackboardSystem shared state."""
    blackboard = BlackboardSystem()
    blackboard.write("key", "value")
    result = blackboard.read("key")
    assert result == "value"

@pytest.mark.asyncio
async def test_group_chat_manager():
    """Test GroupChatManager with mocked agents."""
    agent1 = MagicMock(spec=BaseTemplateAgent)
    agent2 = MagicMock(spec=BaseTemplateAgent)
    agent1._call_llm.return_value = "Response1"
    agent2._call_llm.return_value = "Response2"
    manager = GroupChatManager([agent1, agent2])
    state = await manager.moderate("Test topic")
    assert state.output.startswith("Summarize")
    assert len(state.history) > 0

# Run tests if executed directly
if __name__ == "__main__":
    pytest.main()