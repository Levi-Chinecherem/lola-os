# Standard imports
import pytest
import typing as tp

# Local
from lola.guardrails import ContentSafetyValidator, PIIRedactor, ToolPermissionManager, PromptShield
from lola.agnostic.unified import UnifiedModelManager
from lola.tools.human_input import HumanInputTool

"""
File: Tests for guardrails module in LOLA OS TMVP 1 Phase 2.

Purpose: Verifies guardrail component initialization and functionality.
How: Uses pytest to test guardrail classes.
Why: Ensures safe agent operations, per Radical Reliability.
Full Path: lola-os/tests/test_guardrails.py
"""
@pytest.mark.asyncio
async def test_guardrails_functionality():
    """Test guardrail component functionality."""
    model_manager = UnifiedModelManager()
    validator = ContentSafetyValidator(model_manager)
    redactor = PIIRedactor()
    permissions = ToolPermissionManager()
    shield = PromptShield(model_manager)

    assert isinstance(await validator.validate("test"), dict)
    assert isinstance(redactor.redact("test"), dict)
    assert isinstance(permissions.check_permission(HumanInputTool(), "user"), bool)
    assert isinstance(await shield.shield("test"), dict)