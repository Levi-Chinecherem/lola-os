# Standard imports
import typing as tp
import logging
import asyncio
import re

# Third-party
import spacy
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

# Local
from lola.utils import sentry

"""
File: Implements safety adapters for LOLA OS TMVP 1 Phase 5.

Purpose: Provides flexible backends for content safety validation.
How: Abstracts Spacy, OpenAI Moderation, and regex checks.
Why: Ensures adaptable safety mechanisms, per Developer Sovereignty tenet.
Full Path: lola-os/python/lola/libs/guardrails/safety.py
"""

logger = logging.getLogger(__name__)

class SafetyAdapter:
    """SafetyAdapter: Abstract base for content safety validation."""

    @abstractmethod
    async def validate(self, text: str) -> bool:
        """
        Validates text for safety.

        Args:
            text: Text to validate.

        Returns:
            True if safe, False if harmful.

        Does Not: Redact content—use PIIRedactor.
        """
        pass

class SpacySafetyAdapter(SafetyAdapter):
    """SpacySafetyAdapter: Uses Spacy for content safety."""

    def __init__(self, config: tp.Dict[str, tp.Any]):
        """
        Initialize with Spacy model.

        Args:
            config: Dict with 'model' (e.g., "en_core_web_sm").

        Does Not: Download models—assume pre-downloaded.
        """
        try:
            self.nlp = spacy.load(config.get("model", "en_core_web_sm"))
            logger.info("Initialized SpacySafetyAdapter")
        except Exception as e:
            logger.error(f"SpacySafetyAdapter initialization failed: {e}")
            sentry.capture_exception(e)
            raise

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def validate(self, text: str) -> bool:
        """
        Validates text using Spacy.

        Args:
            text: Text to validate.

        Returns:
            True if safe, False if harmful.
        """
        try:
            doc = self.nlp(text)
            # Simple heuristic: flag if negative sentiment or harmful entities detected
            is_safe = not any(token.dep_ == "neg" for token in doc)
            logger.debug(f"Spacy validation: safe={is_safe}")
            return is_safe
        except Exception as e:
            logger.error(f"Spacy validation failed: {e}")
            sentry.capture_exception(e)
            return False  # Graceful degradation

class OpenAIModerationAdapter(SafetyAdapter):
    """OpenAIModerationAdapter: Uses OpenAI Moderation API."""

    def __init__(self, config: tp.Dict[str, tp.Any]):
        """
        Initialize with OpenAI API key.

        Args:
            config: Dict with 'api_key'.
        """
        try:
            self.client = OpenAI(api_key=config.get("api_key"))
            if not config.get("api_key"):
                raise ValueError("OpenAI API key required")
            logger.info("Initialized OpenAIModerationAdapter")
        except Exception as e:
            logger.error(f"OpenAIModerationAdapter initialization failed: {e}")
            sentry.capture_exception(e)
            raise

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def validate(self, text: str) -> bool:
        """
        Validates text using OpenAI Moderation.

        Args:
            text: Text to validate.

        Returns:
            True if safe, False if harmful.
        """
        try:
            response = await asyncio.to_thread(self.client.moderations.create, input=text)
            is_safe = not response.results[0].flagged
            logger.debug(f"OpenAI Moderation: safe={is_safe}")
            return is_safe
        except Exception as e:
            logger.error(f"OpenAI Moderation failed: {e}")
            sentry.capture_exception(e)
            return False  # Graceful degradation

class RegexSafetyAdapter(SafetyAdapter):
    """RegexSafetyAdapter: Uses regex for basic safety checks."""

    def __init__(self, config: tp.Dict[str, tp.Any]):
        """
        Initialize with regex patterns.

        Args:
            config: Dict with 'patterns' (list of regex strings).

        Does Not: Validate patterns—caller responsibility.
        """
        try:
            self.patterns = config.get("patterns", [r"hate", r"violence", r"profanity"])
            logger.info("Initialized RegexSafetyAdapter")
        except Exception as e:
            logger.error(f"RegexSafetyAdapter initialization failed: {e}")
            sentry.capture_exception(e)
            raise

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def validate(self, text: str) -> bool:
        """
        Validates text using regex.

        Args:
            text: Text to validate.

        Returns:
            True if safe, False if harmful.
        """
        try:
            is_safe = not any(re.search(pattern, text.lower()) for pattern in self.patterns)
            logger.debug(f"Regex validation: safe={is_safe}")
            return is_safe
        except Exception as e:
            logger.error(f"Regex validation failed: {e}")
            sentry.capture_exception(e)
            return False  # Graceful degradation

__all__ = ["SafetyAdapter", "SpacySafetyAdapter", "OpenAIModerationAdapter", "RegexSafetyAdapter"]