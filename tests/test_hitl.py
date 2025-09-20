# Standard imports
import pytest
import typing as tp

# Local
from lola.hitl import ApprovalGatewayNode, EscalationHandler, InteractiveCorrections
from lola.tools.human_input import HumanInputTool
from lola.core.state import State

"""
File: Tests for hitl module in LOLA OS TMVP 1 Phase 2.

Purpose: Verifies human-in-the-loop component initialization and functionality.
How: Uses pytest to test HITL classes.
Why: Ensures robust human interaction, per Choice by Design.
Full Path: lola-os/tests/test_hitl.py
"""
@pytest.mark.asyncio
async def test_hitl_functionality():
    """Test HITL component functionality."""
    human_input = HumanInputTool()
    approval = ApprovalGatewayNode(human_input)
    escalation = EscalationHandler(human_input)
    corrections = InteractiveCorrections(human_input)

    assert isinstance(await approval._approve(State()), dict)
    assert isinstance(await escalation.escalate("test"), dict)
    assert isinstance(await corrections.correct(State(), "test"), dict)