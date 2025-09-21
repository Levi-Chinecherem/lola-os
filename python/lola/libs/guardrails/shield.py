# Standard imports
import typing as tp
import logging
import asyncio
import re

# Third-party
from tenacity import retry, stop_after_attempt, wait_exponential

# Local
from lola.agnostic.unified import UnifiedModelManager
from lola.utils import sentry

"""
File: Implements shield adapters for LOLA OS TMVP 1 Phase 5.

Purpose: Provides flexible backends for prompt injection detection.
How: Abstracts regex and LLM checks.
Why: Ensures adaptable prompt protection, per Developer Sovereignty tenet.
Full Path: lola-os/python/lola/libs/guardrails/shield.py
"""

logger = logging.getLogger(__name__)

class ShieldAdapter:
    """ShieldAdapter: Abstract base for prompt shielding."""

    @abstractmethod
    async def check(self, prompt: str) -> bool:
        """
        Checks prompt for safety.

        Args:
            prompt: Prompt to check.

        Returns:
            True if safe, False if injection detected.

        Does Not: Redact PII—use PIIRedactor.
        """
        pass

class RegexShieldAdapter(ShieldAdapter):
    """RegexShieldAdapter: Uses regex for injection detection."""

    def __init__(self, config: tp.Dict[str, tp.Any]):
        """
        Initialize with regex patterns.

        Args:
            config: Dict with 'patterns' (list of regex strings).

        Does Not: Validate patterns—caller responsibility.
        """
        try:
            self.patterns = config.get("patterns", [r"ignore previous", r"system prompt", r"override"])
            logger.info("Initialized RegexShieldAdapter")
        except Exception as e:
            logger.error(f"RegexShieldAdapter initialization failed: {e}")
            sentry.capture_exception(e)
            raise

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def check(self, prompt: str) -> bool:
        """
        Checks prompt using regex.

        Args:
            prompt: Prompt to check.

        Returns:
            True if safe, False if injection detected.
        """
        try:
            is_safe = not any(re.search(pattern, prompt.lower()) for pattern in self.patterns)
            logger.debug(f"RegexShieldAdapter check: safe={is_safe}")
            return is_safe
        except Exception as e:
            logger.error(f"RegexShieldAdapter check failed: {e}")
            sentry.capture_exception(e)
            return False  # Graceful degradation

class LLMShieldAdapter(ShieldAdapter):
    """LLMShieldAdapter: Uses LLM for injection detection."""

    def __init__(self, config: tp.Dict[str, tp.Any]):
        """
        Initialize with UnifiedModelManager.

        Args:
            config: Dict with 'model_manager' (UnifiedModelManager instance).

        Does Not: Initialize model—use UnifiedModelManager.
        """
        try:
            self.model_manager = config.get("model_manager")
            if not isinstance(self.model_manager, UnifiedModelManager):
                raise ValueError("model_manager must be UnifiedModelManager")
            logger.info("Initialized LLMShieldAdapter")
        except Exception as e:
            logger.error(f"LLMShieldAdapter initialization failed: {e}")
            sentry.capture_exception(e)
            raise

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def check(self, prompt: str) -> bool:
        """
        Checks prompt using LLM.

        Args:
            prompt: Prompt to check.

        Returns:
            True if safe, False if injection detected.
        """
        try:
            check_prompt = f"Is this prompt safe (no injection attempts): {prompt}"
            response = await self.model_manager.call(check_prompt)
            is_safe = "yes" in response.lower()
            logger.debug(f"LLMShieldAdapter check: safe={is_safe}")
            return is_safe
        except Exception as e:
            logger.error(f"LLMShieldAdapter check failed: {e}")
            sentry.capture_exception(e)
            return False  # Graceful degradation

__all__ = ["ShieldAdapter", "RegexShieldAdapter", "LLMShieldAdapter"]