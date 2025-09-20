# Standard imports
import typing as tp

# Third-party
import litellm

"""
File: Defines the UnifiedModelManager for LOLA OS TMVP 1.

Purpose: Provides a single interface for LLM calls across providers.
How: Wraps litellm's completion API for model-agnostic access.
Why: Simplifies model switching without code changes, per Agnostic Adapter Pattern.
Full Path: lola-os/python/lola/agnostic/unified.py
"""

class UnifiedModelManager:
    """Unified interface for LLM calls using litellm."""

    def __init__(self, default_model: str = "openai/gpt-4o"):
        """
        Initialize with a default LLM model.

        Args:
            default_model: Default model string for litellm (e.g., "openai/gpt-4o").
        """
        self.default_model = default_model

    async def complete(self, prompt: str, model: tp.Optional[str] = None) -> str:
        """
        Execute an LLM completion.

        Args:
            prompt: Input prompt string.
            model: Optional model string to override default.
        Returns:
            str: LLM response.
        """
        model = model or self.default_model
        response = await litellm.acompletion(
            model=model,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content