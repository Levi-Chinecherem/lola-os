# Standard imports
import typing as tp

# Third-party
from litellm import completion

"""
File: Defines the UnifiedModelManager class for LOLA OS TMVP 1 Phase 2.

Purpose: Provides a unified interface for LLM calls across providers.
How: Wraps litellm for real model switching and calls.
Why: Ensures LLM agnosticism, per Developer Sovereignty tenet.
Full Path: lola-os/python/lola/agnostic/unified.py
Future Optimization: Migrate to Rust for fast LLM routing (post-TMVP 1).
"""

class UnifiedModelManager:
    """UnifiedModelManager: Manages LLM calls. Does NOT persist responses—use StateManager."""

    async def call(self, prompt: str, model: str = "openai/gpt-4o") -> str:
        """
        Calls LLM with prompt.

        Args:
            prompt: Input prompt.
            model: LLM model string.

        Returns:
            Response string.

        Does Not: Handle cost—use cost.py.
        """
        response = completion(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1000
        )
        return response.choices[0].message.content

__all__ = ["UnifiedModelManager"]