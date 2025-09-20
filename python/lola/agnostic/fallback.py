# Standard imports
import typing as tp

# Third-party
import litellm

"""
File: Defines the ModelFallbackBalancer for LOLA OS TMVP 1.

Purpose: Handles automatic retries and fallback for LLM calls.
How: Uses litellm's fallback mechanism for failed requests.
Why: Ensures robust LLM interactions, per Radical Reliability.
Full Path: lola-os/python/lola/agnostic/fallback.py
Future Optimization: Migrate to Rust for high-throughput load balancing (post-TMVP 1).
"""

class ModelFallbackBalancer:
    """Manages fallback and load balancing for LLM calls."""

    def __init__(self, models: tp.List[str]):
        """
        Initialize with a list of fallback models.

        Args:
            models: List of model strings (e.g., ["openai/gpt-4o", "anthropic/claude-3-sonnet"]).
        """
        self.models = models

    async def complete(self, prompt: str) -> str:
        """
        Execute an LLM completion with fallback.

        Args:
            prompt: Input prompt string.
        Returns:
            str: LLM response from successful model.
        """
        for model in self.models:
            try:
                response = await litellm.acompletion(
                    model=model,
                    messages=[{"role": "user", "content": prompt}]
                )
                return response.choices[0].message.content
            except Exception:
                continue
        raise RuntimeError("All fallback models failed")