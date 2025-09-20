"""
File: Initializes the specialization module for LOLA OS TMVP 1 Phase 2.

Purpose: Exports specialization components for agents.
How: Defines package-level exports for specialization tools.
Why: Centralizes access to fine-tuning and skills, per Developer Sovereignty.
Full Path: lola-os/python/lola/specialization/__init__.py
"""
from .fine_tuning import FineTuningHarness
from .skill_library import SkillLibrary

__all__ = ["FineTuningHarness", "SkillLibrary"]