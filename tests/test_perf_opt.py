# Standard imports
import os
import pytest
import asyncio
from unittest.mock import patch

# Local
from lola.perf_opt.prompt_compressor import PromptCompressor
from lola.perf_opt.caching import CachingLayer
from lola.perf_opt.hardware import HardwareOptimizer
from lola.agnostic.unified import UnifiedModelManager

"""
File: Comprehensive tests for LOLA OS performance optimizations in Phase 2.

Purpose: Validates prompt compression, caching, and hardware optimization with real Redis and mocked LLM.
How: Uses pytest with async support, patch for LLM, Redis test instance for caching.
Why: Ensures efficient execution with >80% coverage, per Radical Reliability tenet.
Full Path: lola-os/tests/test_perf_opt.py
"""

@pytest.mark.asyncio
async def test_prompt_compressor(mocker):
    """Test PromptCompressor with mocked LLM."""
    mocker.patch('lola.agnostic.unified.UnifiedModelManager.call', return_value="Compressed prompt")
    manager = UnifiedModelManager("test/model")
    compressor = PromptCompressor(manager)
    result = await compressor.compress("Long prompt to compress")
    assert result == "Compressed prompt"

@pytest.mark.asyncio
async def test_caching_layer(redis):
    """Test CachingLayer with real Redis instance."""
    layer = CachingLayer("redis://localhost:6379/0")
    await layer.set("test_key", "test_value")
    result = await layer.get("test_key")
    assert result == "test_value"

def test_hardware_optimizer(mocker):
    """Test HardwareOptimizer with mocked os."""
    mocker.patch('os.system', return_value=0)
    optimizer = HardwareOptimizer()
    optimizer.optimize()
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "0"
    assert os.environ.get("USE_FLASH_ATTENTION") == "1"

# Run tests if executed directly
if __name__ == "__main__":
    pytest.main()