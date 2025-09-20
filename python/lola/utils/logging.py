# Standard imports
import logging
import sys
from typing import Optional

"""
File: Structured logging setup for LOLA OS TMVP 1 Phase 3.

Purpose: Configures logging with structured output for debugging and monitoring.
How: Uses Python's logging module with JSON-like formatting.
Why: Enables traceable diagnostics, per Radical Reliability.
Full Path: lola-os/python/lola/utils/logging.py
"""

logger = logging.getLogger("lola-os")

def setup_logging(level: str = "INFO", output: Optional[str] = None) -> None:
    """
    Set up structured logging.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR).
        output: Optional file path for log output (defaults to stdout).
    """
    log_level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(log_level)

    formatter = logging.Formatter(
        '{"time": "%(asctime)s", "level": "%(levelname)s", "message": "%(message)s"}'
    )

    if output:
        handler = logging.FileHandler(output)
    else:
        handler = logging.StreamHandler(sys.stdout)

    handler.setLevel(log_level)  # Explicitly set handler level
    handler.setFormatter(formatter)
    logger.handlers = [handler]