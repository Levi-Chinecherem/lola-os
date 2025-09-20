# Standard imports
import pytest
import typing as tp

# Local
from lola.orchestration import AgentSwarmOrchestrator, ContractNetProtocol, BlackboardSystem, GroupChatManager
from lola.agents.react import ReActAgent
from lola.tools.human_input import HumanInputTool
from lola.core.state import State

"""
File: Tests for orchestration module in LOLA OS TMVP 1 Phase 2.

Purpose: Verifies orchestration component initialization and functionality.
How: Uses pytest to test orchestration classes.
Why: Ensures robust multi-agent coordination, per Choice by Design.
Full Path: lola-os/tests/test_orchestration.py
"""
@pytest.mark.asyncio
async def test_orchestration_functionality():
    """Test orchestration component functionality."""
    agent = ReActAgent(tools=[HumanInputTool()], model="openai/gpt-4o")
    swarm = AgentSwarmOrchestrator([agent])
    contract_net = ContractNetProtocol([agent])
    blackboard = BlackboardSystem()
    group_chat = GroupChatManager([agent])

    assert isinstance(await swarm.orchestrate("test"), dict)
    assert isinstance(await contract_net.allocate_task("test"), dict)
    assert blackboard.read("test") is not None
    assert isinstance(await group_chat.moderate("test"), dict)