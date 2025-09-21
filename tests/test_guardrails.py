# Standard imports
import pytest
import asyncio
from unittest.mock import AsyncMock

# Local
from lola.guardrails.content_safety import ContentSafetyValidator
from lola.guardrails.pii_redactor import PIIRedactor
from lola.guardrails.tool_permissions import ToolPermissionManager
from lola.guardrails.prompt_shield import PromptShield
from lola.agnostic.unified import UnifiedModelManager
from lola.tools.base import BaseTool

"""
File: Comprehensive tests for LOLA OS guardrails in Phase 2.

Purpose: Validates safety features with real regex and mocked LLM calls.
How: Uses pytest with async support, test data for validation.
Why: Ensures secure operations with >80% coverage, per Radical Reliability tenet.
Full Path: lola-os/tests/test_guardrails.py
"""

@pytest.mark.asyncio
async def test_content_safety_validator(mocker):
    """Test ContentSafetyValidator with mocked LLM."""
    mocker.patch('lola.agnostic.unified.UnifiedModelManager.call', return_value="yes")
    manager = UnifiedModelManager("test/model")
    validator = ContentSafetyValidator(manager)
    result = await validator.validate("safe text")
    assert result is True
    result = await validator.validate("hate speech")
    assert result is False

def test_pii_redactor():
    """Test PIIRedactor with sample text."""
    redactor = PIIRedactor()
    text = "Email: test@example.com, Phone: 123-456-7890"
    result = redactor.redact(text)
    assert "test@example.com" not in result
    assert "123-456-7890" not in result

def test_tool_permission_manager():
    """Test ToolPermissionManager with roles and tools."""
    manager = ToolPermissionManager()
    manager.add_permission("user", "safe_tool")
    tool = BaseTool()
    tool.name = "safe_tool"
    assert manager.check_permission("user", tool) is True
    tool.name = "restricted_tool"
    assert manager.check_permission("user", tool) is False

@pytest.mark.asyncio
async def test_prompt_shield(mocker):
    """Test PromptShield with mocked LLM."""
    mocker.patch('lola.agnostic.unified.UnifiedModelManager.call', return_value="yes")
    manager = UnifiedModelManager("test/model")
    shield = PromptShield(manager)
    result = await shield.shield("safe prompt")
    assert result is True
    result = await shield.shield("ignore previous")
    assert result is False

# Run tests if executed directly
if __name__ == "__main__":
    pytest.main()