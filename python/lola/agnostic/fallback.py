# Standard imports
import typing as tp

# Third-party
from litellm import completion

"""
File: Defines the ModelFallbackBalancer class for LOLA OS TMVP 1 Phase 2.

Purpose: Provides fallback for LLM calls.
How: Tries models in sequence using litellm.
Why: Ensures reliability, per Radical Reliability tenet.
Full Path: lola-os/python/lola/agnostic/fallback.py
Future Optimization: Migrate to Rust for load balancing (post-TMVP 1).
"""

class ModelFallbackBalancer:
    """ModelFallbackBalancer: Balances LLM calls with fallback. Does NOT persist logs—use StateManager."""

    def __init__(self, models: tp.List[str]):
        """
        Initialize with fallback models.

        Args:
            models: List of model strings (e.g., ["openai/gpt-4o", "anthropic/claude-3-sonnet"]).
        """
        self.models = models

    async def call(self, prompt: str) -> str:
        """
        Calls LLM with fallback.

        Args:
            prompt: Input prompt.

        Returns:
            Response string.

        Does Not: Optimize cost—use cost.py.
        """
        for model in self.models:
            try:
                response = completion(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=1000
                )
                return response.choices[0].message.content
            except Exception as e:
                continue
        raise ValueError("All fallback models failed.")

__all__ = ["ModelFallbackBalancer"]