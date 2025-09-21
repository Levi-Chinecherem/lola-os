# Standard imports
import pytest
import asyncio
from unittest.mock import AsyncMock, patch, MagicMock

# Local
from lola.guardrails.content_safety import ContentSafetyValidator
from lola.guardrails.pii_redactor import PIIRedactor
from lola.guardrails.tool_permissions import ToolPermissionManager
from lola.guardrails.prompt_shield import PromptShield
from lola.agnostic.unified import UnifiedModelManager
from lola.tools.base import BaseTool
from lola.utils import sentry
from lola.utils import prometheus

"""
File: Comprehensive tests for LOLA OS guardrails in Phase 5.

Purpose: Validates safety features with real backends (Spacy, Presidio) and mocks for APIs.
How: Uses pytest with async support, test data for validation, and integration tests.
Why: Ensures secure operations with >80% coverage, per Radical Reliability tenet.
Full Path: lola-os/tests/test_guardrails.py
"""

@pytest.fixture
def mock_model_manager():
    """Fixture for mocked UnifiedModelManager."""
    return MagicMock(spec=UnifiedModelManager, call=AsyncMock(return_value="yes"))

@pytest.mark.asyncio
async def test_content_safety_validator_spacy(mocker):
    """Test ContentSafetyValidator with Spacy backend."""
    mocker.patch("lola.utils.sentry.init_sentry")
    mocker.patch("prometheus_client.Counter.inc")
    config = {"safety_backend": {"type": "spacy", "model": "en_core_web_sm"}}
    validator = ContentSafetyValidator(config)
    result = await validator.validate("safe text")
    assert result is True

@pytest.mark.asyncio
async def test_pii_redactor_presidio(mocker):
    """Test PIIRedactor with Presidio backend."""
    mocker.patch("lola.utils.sentry.init_sentry")
    mocker.patch("prometheus_client.Counter.inc")
    config = {"pii_backend": {"type": "presidio"}}
    redactor = PIIRedactor(config)
    text = "Email: test@example.com, Phone: 123-456-7890"
    result = await redactor.redact(text)
    assert "test@example.com" not in result
    assert "123-456-7890" not in result

def test_tool_permission_manager():
    """Test ToolPermissionManager with config."""
    config = {"roles": {"user": ["safe_tool"], "admin": ["safe_tool", "admin_tool"]}}
    manager = ToolPermissionManager(config)
    tool = BaseTool()
    tool.name = "safe_tool"
    assert manager.check_permission("user", tool) is True
    tool.name = "admin_tool"
    assert manager.check_permission("user", tool) is False

@pytest.mark.asyncio
async def test_prompt_shield_llm(mocker, mock_model_manager):
    """Test PromptShield with LLM backend."""
    mocker.patch("lola.utils.sentry.init_sentry")
    mocker.patch("prometheus_client.Counter.inc")
    config = {"shield_backend": {"type": "llm"}}
    shield = PromptShield(config)
    result = await shield.shield("safe prompt")
    assert result is True
    result = await shield.shield("ignore previous")
    assert result is False

# Run tests if executed directly
if __name__ == "__main__":
    pytest.main()