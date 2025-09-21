# Standard imports
import logging
import sys
from logging.handlers import RotatingFileHandler
import json

"""
File: Structured logging for LOLA OS TMVP 1 Phase 3.

Purpose: Provides JSON-formatted logging for diagnostics.
How: Uses Python's logging with a custom JSON formatter.
Why: Enables traceable diagnostics, per Radical Reliability.
Full Path: lola-os/python/lola/utils/logging.py
"""

class JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
            "funcName": record.funcName,
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
        }
        return json.dumps(log_data)

logger = logging.getLogger("lola")
logger.setLevel(logging.INFO)

def setup_logging(verbose: bool = False) -> None:
    """
    Initialize logging with JSON format and optional verbosity.

    Args:
        verbose: If True, set logging level to DEBUG.

    Does not:
        Modify external logging configurations.
    """
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(JSONFormatter())
    logger.addHandler(console_handler)

    # File handler with rotation
    file_handler = RotatingFileHandler("lola.log", maxBytes=10_000_000, backupCount=5)
    file_handler.setFormatter(JSONFormatter())
    logger.addHandler(file_handler)