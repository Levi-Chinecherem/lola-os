# Third-party
from pydantic import BaseModel
from typing import Dict, Any

"""
File: Defines the State class for LOLA OS TMVP 1.

Purpose: Provides a strongly-typed structure for agent state.
How: Uses Pydantic for validation and serialization.
Why: Ensures predictable state management, per Radical Reliability.
Full Path: lola-os/python/lola/core/state.py
"""

class State(BaseModel):
    """Strongly-typed Pydantic model for agent state."""

    data: Dict[str, Any] = {}
    history: list = []

    def update(self, new_data: Dict[str, Any]) -> None:
        """
        Update state with new data and append to history.

        Args:
            new_data: Dictionary of new state values.
        """
        self.data.update(new_data)
        self.history.append(new_data)