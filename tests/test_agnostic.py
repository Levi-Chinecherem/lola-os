# Standard imports
import pytest
import asyncio
from unittest.mock import patch

# Local
from lola.agnostic.unified import UnifiedModelManager
from lola.agnostic.fallback import ModelFallbackBalancer
from lola.agnostic.cost import CostOptimizer

"""
File: Comprehensive tests for LOLA OS LLM-agnostic in Phase 2.

Purpose: Validates provider switching, fallback, and cost optimization with mocked litellm.
How: Uses pytest with async support, patch for litellm calls.
Why: Ensures flexible LLM usage with >80% coverage, per Developer Sovereignty tenet.
Full Path: lola-os/tests/test_agnostic.py
"""

@pytest.mark.asyncio
async def test_unified_model_manager(mocker):
    """Test UnifiedModelManager with mocked litellm."""
    mocker.patch('litellm.completion', return_value={"choices": [{"message": {"content": "Test response"}}]})
    manager = UnifiedModelManager("test/model")
    result = await manager.call("Test prompt")
    assert result == "Test response"

@pytest.mark.asyncio
async def test_model_fallback_balancer(mocker):
    """Test ModelFallbackBalancer with mocked litellm."""
    mocker.patch('litellm.completion', side_effect=[Exception("Fail"), {"choices": [{"message": {"content": "Fallback response"}}]}])
    balancer = ModelFallbackBalancer(["fail/model", "success/model"])
    result = await balancer.call("Test prompt")
    assert result == "Fallback response"

@pytest.mark.asyncio
async def test_cost_optimizer(mocker):
    """Test CostOptimizer with mocked litellm."""
    mocker.patch('litellm.completion', return_value={"choices": [{"message": {"content": "Cheap response"}}]})
    optimizer = CostOptimizer({"cheap/model": 0.1, "expensive/model": 1.0})
    result = await optimizer.call("Test prompt", complexity="low")
    assert result == "Cheap response"

# Run tests if executed directly
if __name__ == "__main__":
    pytest.main()