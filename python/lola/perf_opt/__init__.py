"""
File: Initializes the perf_opt module for LOLA OS TMVP 1 Phase 2.

Purpose: Exports performance optimization components.
How: Defines package-level exports for optimization tools.
Why: Centralizes access to performance utilities, per Radical Reliability.
Full Path: lola-os/python/lola/perf_opt/__init__.py
"""
from .prompt_compressor import PromptCompressor
from .caching import CachingLayer
from .hardware import HardwareOptimizer

__all__ = ["PromptCompressor", "CachingLayer", "HardwareOptimizer"]