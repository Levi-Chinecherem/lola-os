# Standard imports
import typing as tp
import json

# Third-party
from pydantic import BaseModel

# Local
from .state import State

"""
File: Defines memory management classes for LOLA OS TMVP 1.

Purpose: Manages state persistence, conversation history, and entity extraction.
How: Implements StateManager, ConversationMemory, and EntityMemory with JSON persistence.
Why: Enables reliable state and context retention, per Developer Sovereignty.
Full Path: lola-os/python/lola/core/memory.py
Future Optimization: Migrate to Rust for high-throughput persistence (post-TMVP 1).
"""

class ConversationMemory(BaseModel):
    """Manages dialog history with LLMs."""

    history: tp.List[tp.Dict[str, str]] = []

    def add_message(self, role: str, content: str) -> None:
        """
        Add a message to the conversation history.

        Args:
            role: Message role (e.g., "user", "assistant").
            content: Message content.
        """
        self.history.append({"role": role, "content": content})

class EntityMemory(BaseModel):
    """Manages extracted entities (people, places, facts)."""

    entities: tp.Dict[str, tp.Any] = {}

    def add_entity(self, key: str, value: tp.Any) -> None:
        """
        Add an extracted entity to memory.

        Args:
            key: Entity identifier (e.g., "user_name").
            value: Entity value.
        """
        self.entities[key] = value

class StateManager:
    """Handles persistence, checkpointing, and loading of state."""

    def __init__(self, storage_path: str = "state.json"):
        """
        Initialize the state manager.

        Args:
            storage_path: Path for JSON state persistence.
        """
        self.storage_path = storage_path
        self.conversation = ConversationMemory()
        self.entities = EntityMemory()

    async def checkpoint(self, state: State) -> None:
        """
        Persist the current state to storage.

        Args:
            state: State object to persist.
        """
        with open(self.storage_path, "w") as f:
            json.dump(state.dict(), f)

    async def load(self) -> State:
        """
        Load state from storage.

        Returns:
            State: Loaded state object.
        """
        try:
            with open(self.storage_path, "r") as f:
                return State(**json.load(f))
        except FileNotFoundError:
            return State()