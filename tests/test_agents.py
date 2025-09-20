# Standard imports
import pytest
import typing as tp

# Local
from lola.agents import (
    ReActAgent, PlanExecuteAgent, CollaborativeAgent, RouterAgent,
    ConversationalAgent, CodingAgent, OrchestratorAgent, MetaCognitionAgent,
    SyntheticDataAgent, LegacyInterfaceAgent
)
from lola.tools.human_input import HumanInputTool
from lola.core.state import State

"""
File: Tests for agents module in LOLA OS TMVP 1 Phase 2.

Purpose: Verifies agent initialization and basic functionality.
How: Uses pytest to test agent classes.
Why: Ensures robust agent implementations, per Radical Reliability.
Full Path: lola-os/tests/test_agents.py
"""
@pytest.mark.asyncio
async def test_agent_initialization():
    """Test initialization of all agents."""
    tools = [HumanInputTool()]
    agents = [
        ReActAgent(tools),
        PlanExecuteAgent(tools),
        CollaborativeAgent(tools),
        RouterAgent(tools),
        ConversationalAgent(tools),
        CodingAgent(tools),
        OrchestratorAgent(tools),
        MetaCognitionAgent(tools),
        SyntheticDataAgent(tools),
        LegacyInterfaceAgent(tools),
    ]
    for agent in agents:
        assert isinstance(agent, ReActAgent) or isinstance(agent, PlanExecuteAgent) or \
               isinstance(agent, CollaborativeAgent) or isinstance(agent, RouterAgent) or \
               isinstance(agent, ConversationalAgent) or isinstance(agent, CodingAgent) or \
               isinstance(agent, OrchestratorAgent) or isinstance(agent, MetaCognitionAgent) or \
               isinstance(agent, SyntheticDataAgent) or isinstance(agent, LegacyInterfaceAgent)
        result = await agent.run("test query")
        assert isinstance(result, State)