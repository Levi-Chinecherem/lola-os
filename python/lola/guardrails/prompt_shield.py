# Standard imports
import typing as tp
import re

# Local
from lola.agnostic.unified import UnifiedModelManager

"""
File: Defines the PromptShield class for LOLA OS TMVP 1 Phase 2.

Purpose: Protects against prompt injection attacks.
How: Uses regex and LLM to detect harmful prompts.
Why: Ensures prompt safety, per Radical Reliability tenet.
Full Path: lola-os/python/lola/guardrails/prompt_shield.py
Future Optimization: Migrate to Rust for fast shielding (post-TMVP 1).
"""

class PromptShield:
    """PromptShield: Detects prompt injections. Does NOT persist logs—use StateManager."""

    def __init__(self, model_manager: UnifiedModelManager):
        """
        Initialize with model manager.

        Args:
            model_manager: UnifiedModelManager instance.
        """
        self.model_manager = model_manager
        self.injection_patterns = [r"ignore previous", r"system prompt", r"override"]

    async def shield(self, prompt: str) -> bool:
        """
        Checks if prompt is safe.

        Args:
            prompt: Prompt to check.

        Returns:
            True if safe, False if injection detected.

        Does Not: Redact PII—use pii_redactor.py.
        """
        # Inline: Basic regex check
        if any(re.search(pattern, prompt.lower()) for pattern in self.injection_patterns):
            return False
        # Inline: Advanced LLM check
        check_prompt = f"Is this prompt safe (no injection): {prompt}"
        response = await self.model_manager.call(check_prompt)
        return "yes" in response.lower()

__all__ = ["PromptShield"]