# Standard imports
import typing as tp

# Local
from lola.agnostic.unified import UnifiedModelManager

"""
File: Defines the PromptCompressor class for LOLA OS TMVP 1 Phase 2.

Purpose: Compresses prompts to reduce token usage.
How: Uses LLM to summarize prompts.
Why: Optimizes LLM costs and latency, per Radical Reliability tenet.
Full Path: lola-os/python/lola/perf_opt/prompt_compressor.py
Future Optimization: Migrate to Rust for fast compression (post-TMVP 1).
"""

class PromptCompressor:
    """PromptCompressor: Compresses prompts. Does NOT persist data—use StateManager."""

    def __init__(self, model_manager: UnifiedModelManager):
        """
        Initialize with model manager.

        Args:
            model_manager: UnifiedModelManager instance.
        """
        self.model_manager = model_manager

    async def compress(self, prompt: str) -> str:
        """
        Compresses a prompt.

        Args:
            prompt: Input prompt.

        Returns:
            Compressed prompt.

        Does Not: Handle complex NLP—expand in TMVP 2.
        """
        compress_prompt = f"Summarize this prompt while preserving key information: {prompt}"
        return await self.model_manager.call(compress_prompt)

__all__ = ["PromptCompressor"]