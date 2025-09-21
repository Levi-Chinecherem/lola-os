# Standard imports
import typing as tp
import threading

# Local
from lola.core.state import State
from lola.utils.logging import logger

"""
File: Defines the BlackboardSystem class for LOLA OS TMVP 1 Phase 2.

Purpose: Provides shared knowledge space for agents.
How: Uses thread-safe dict for shared state.
Why: Enables collaborative knowledge, per Choice by Design tenet.
Full Path: lola-os/python/lola/orchestration/blackboard.py
Future Optimization: Migrate to Rust for distributed blackboard (post-TMVP 1).
"""

class BlackboardSystem:
    """BlackboardSystem: Shared knowledge space for agents. Does NOT persist data—use StateManager."""

    def __init__(self):
        """
        Initialize blackboard.
        """
        self.blackboard = {}
        self.lock = threading.Lock()
        logger.info("Initialized BlackboardSystem.")

    def write(self, key: str, value: tp.Any) -> None:
        """
        Writes to blackboard.

        Args:
            key: Key name.
            value: Value to write.

        Does Not: Validate value—caller must ensure.
        """
        with self.lock:
            self.blackboard[key] = value

    def read(self, key: str) -> tp.Any:
        """
        Reads from blackboard.

        Args:
            key: Key name.

        Returns:
            Value or None.

        Does Not: Handle missing keys—caller must check.
        """
        with self.lock:
            return self.blackboard.get(key)

__all__ = ["BlackboardSystem"]