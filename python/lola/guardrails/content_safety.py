# Standard imports
import typing as tp
import re

# Local
from lola.agnostic.unified import UnifiedModelManager

"""
File: Defines the ContentSafetyValidator class for LOLA OS TMVP 1 Phase 2.

Purpose: Validates content for toxicity and harm.
How: Uses regex for basic checks, LLM for advanced.
Why: Ensures safe outputs, per Radical Reliability tenet.
Full Path: lola-os/python/lola/guardrails/content_safety.py
Future Optimization: Migrate to Rust for fast validation (post-TMVP 1).
"""

class ContentSafetyValidator:
    """ContentSafetyValidator: Validates content safety. Does NOT persist logs—use StateManager."""

    def __init__(self, model_manager: UnifiedModelManager):
        """
        Initialize with model manager.

        Args:
            model_manager: UnifiedModelManager instance.
        """
        self.model_manager = model_manager
        self.toxic_patterns = [r"hate", r"violence", r"profanity"]

    async def validate(self, text: str) -> bool:
        """
        Validates text for safety.

        Args:
            text: Text to validate.

        Returns:
            True if safe, False if harmful.

        Does Not: Redact PII—use pii_redactor.py.
        """
        # Inline: Basic regex check
        if any(re.search(pattern, text.lower()) for pattern in self.toxic_patterns):
            return False
        # Inline: Advanced LLM check
        prompt = f"Is this text safe (no hate/violence): {text}"
        response = await self.model_manager.call(prompt)
        return "yes" in response.lower()

__all__ = ["ContentSafetyValidator"]