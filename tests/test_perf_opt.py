# Standard imports
import pytest
import typing as tp

# Local
from lola.perf_opt import PromptCompressor, CachingLayer, HardwareOptimizer
from lola.agnostic.unified import UnifiedModelManager

"""
File: Tests for perf_opt module in LOLA OS TMVP 1 Phase 2.

Purpose: Verifies performance optimization component initialization and functionality.
How: Uses pytest to test optimization classes.
Why: Ensures efficient execution, per Radical Reliability.
Full Path: lola-os/tests/test_perf_opt.py
"""
@pytest.mark.asyncio
async def test_perf_opt_functionality():
    """Test performance optimization component functionality."""
    model_manager = UnifiedModelManager()
    compressor = PromptCompressor(model_manager)
    caching = CachingLayer()
    optimizer = HardwareOptimizer()

    assert isinstance(await compressor.compress("test"), str)
    assert caching.get("test") is None
    caching.set("test", "value")
    assert caching.get("test") == "value"
    assert isinstance(optimizer.optimize({}), dict)