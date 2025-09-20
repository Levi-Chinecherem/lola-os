# Standard imports
import pytest
import typing as tp

# Local
from lola.specialization import FineTuningHarness, SkillLibrary
from lola.agnostic.unified import UnifiedModelManager

"""
File: Tests for specialization module in LOLA OS TMVP 1 Phase 2.

Purpose: Verifies specialization component initialization and functionality.
How: Uses pytest to test specialization classes.
Why: Ensures customized agent capabilities, per Developer Sovereignty.
Full Path: lola-os/tests/test_specialization.py
"""
@pytest.mark.asyncio
async def test_specialization_functionality():
    """Test specialization component functionality."""
    model_manager = UnifiedModelManager()
    harness = FineTuningHarness(model_manager)
    library = SkillLibrary()

    assert isinstance(await harness.fine_tune({}), dict)
    library.add_skill("test", lambda: "skill")
    assert library.get_skill("test")() == "skill"