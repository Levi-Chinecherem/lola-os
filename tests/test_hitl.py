# Standard imports
import pytest
import asyncio
from unittest.mock import AsyncMock

# Local
from lola.hitl.approval import ApprovalGatewayNode
from lola.hitl.escalation import EscalationHandler
from lola.hitl.corrections import InteractiveCorrections
from lola.core.state import State
from lola.tools.human_input import HumanInputTool

"""
File: Comprehensive tests for LOLA OS HITL in Phase 2.

Purpose: Validates human-in-the-loop features with mocked input.
How: Uses pytest with async support, AsyncMock for input.
Why: Ensures robust human oversight with >80% coverage, per Choice by Design tenet.
Full Path: lola-os/tests/test_hitl.py
"""

@pytest.mark.asyncio
async def test_approval_gateway_node(mocker):
    """Test ApprovalGatewayNode with mocked input."""
    mocker.patch('lola.tools.human_input.HumanInputTool.execute', return_value="yes")
    node = ApprovalGatewayNode()
    state = State(output="Test action")
    state = await node.approve(state)
    assert state.metadata["approved"] is True

@pytest.mark.asyncio
async def test_escalation_handler(mocker):
    """Test EscalationHandler with mocked input."""
    mocker.patch('lola.tools.human_input.HumanInputTool.execute', return_value="guidance")
    handler = EscalationHandler()
    state = State(query="Test")
    state = await handler.escalate(state, 0.7)
    assert state.output == "guidance"

@pytest.mark.asyncio
async def test_interactive_corrections(mocker):
    """Test InteractiveCorrections with mocked input."""
    mocker.patch('lola.tools.human_input.HumanInputTool.execute', return_value="correction")
    corrections = InteractiveCorrections()
    state = State(output="wrong")
    state = await corrections.correct(state, "error")
    assert state.output == "correction"

# Run tests if executed directly
if __name__ == "__main__":
    pytest.main()