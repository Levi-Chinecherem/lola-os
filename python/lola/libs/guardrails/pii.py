# Standard imports
import typing as tp
import logging
import asyncio
import re

# Third-party
from tenacity import retry, stop_after_attempt, wait_exponential
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine

# Local
from lola.utils import sentry

"""
File: Implements PII adapters for LOLA OS TMVP 1 Phase 5.

Purpose: Provides flexible backends for PII redaction.
How: Abstracts Presidio and regex checks.
Why: Ensures adaptable privacy protection, per Developer Sovereignty tenet.
Full Path: lola-os/python/lola/libs/guardrails/pii.py
"""

logger = logging.getLogger(__name__)

class PIIAdapter:
    """PIIAdapter: Abstract base for PII redaction."""

    @abstractmethod
    async def redact(self, text: str) -> str:
        """
        Redacts PII from text.

        Args:
            text: Text to redact.

        Returns:
            Redacted text.

        Does Not: Validate content—use ContentSafetyValidator.
        """
        pass

class PresidioAdapter(PIIAdapter):
    """PresidioAdapter: Uses Presidio for PII redaction."""

    def __init__(self, config: tp.Dict[str, tp.Any]):
        """
        Initialize with Presidio engines.

        Args:
            config: Dict with 'language' (default "en").
        """
        try:
            self.analyzer = AnalyzerEngine()
            self.anonymizer = AnonymizerEngine()
            self.language = config.get("language", "en")
            logger.info("Initialized PresidioAdapter")
        except Exception as e:
            logger.error(f"PresidioAdapter initialization failed: {e}")
            sentry.capture_exception(e)
            raise

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def redact(self, text: str) -> str:
        """
        Redacts PII using Presidio.

        Args:
            text: Text to redact.

        Returns:
            Redacted text.
        """
        try:
            results = self.analyzer.analyze(text=text, language=self.language, entities=["PERSON", "PHONE_NUMBER", "EMAIL_ADDRESS"])
            redacted = self.anonymizer.anonymize(text=text, analyzer_results=results)
            logger.debug("PII redacted with Presidio")
            return redacted.text
        except Exception as e:
            logger.error(f"Presidio redaction failed: {e}")
            sentry.capture_exception(e)
            return text  # Graceful degradation

class RegexPIIAdapter(PIIAdapter):
    """RegexPIIAdapter: Uses regex for basic PII redaction."""

    def __init__(self, config: tp.Dict[str, tp.Any]):
        """
        Initialize with regex patterns.

        Args:
            config: Dict with 'patterns' (list of regex).
        """
        self.patterns = config.get("patterns", [
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email
            r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'  # Phone
        ])
        logger.info("Initialized RegexPIIAdapter")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def redact(self, text: str) -> str:
        """
        Redacts PII using regex.

        Args:
            text: Text to redact.

        Returns:
            Redacted text.
        """
        try:
            redacted = text
            for pattern in self.patterns:
                redacted = re.sub(pattern, "[PII]", redacted)
            logger.debug("PII redacted with regex")
            return redacted
        except Exception as e:
            logger.error(f"Regex PII redaction failed: {e}")
            sentry.capture_exception(e)
            return text

__all__ = ["PIIAdapter", "PresidioAdapter", "RegexPIIAdapter"]