"""
File: Initializes the agnostic module for LOLA OS TMVP 1.

Purpose: Exports LLM-agnostic components for unified model access.
How: Wraps litellm for model switching and load balancing.
Why: Enables seamless LLM provider switching, per Developer Sovereignty.
Full Path: lola-os/python/lola/agnostic/__init__.py
"""
from .unified import UnifiedModelManager
from .fallback import ModelFallbackBalancer
from .cost import CostOptimizer

__all__ = ["UnifiedModelManager", "ModelFallbackBalancer", "CostOptimizer"]