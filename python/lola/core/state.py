from pydantic import BaseModel
import typing as tp

"""
File: Defines the State Pydantic model for LOLA OS graph execution.

Purpose: Provides a strongly-typed, serializable state for tracking agent execution context.
How: Uses Pydantic for validation and serialization, stores data like query, history, and outputs.
Why: Ensures radical reliability via explicit state management, per LOLA's tenets.
Full Path: lola-os/python/lola/core/state.py
"""

class State(BaseModel):
    """State: Pydantic model for agent execution state."""

    query: str = ""  # Initial user query
    history: tp.List[tp.Dict[str, str]] = []  # Conversation history
    output: tp.Optional[str] = None  # Current node output
    metadata: tp.Dict[str, tp.Any] = {}  # Flexible storage for tools/EVM

    class Config:
        """Pydantic config for JSON serialization."""
        arbitrary_types_allowed = True

    def update(self, **kwargs) -> None:
        """
        Updates state fields with validation.

        Args:
            **kwargs: Fields to update (e.g., query, output).

        Does Not: Persist to storage—use StateManager.
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

__all__ = ["State"]