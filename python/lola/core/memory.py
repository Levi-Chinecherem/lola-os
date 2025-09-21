import json
import typing as tp
from pathlib import Path
import litellm
from pydantic import BaseModel
from .state import State

"""
File: Implements StateManager, ConversationMemory, and EntityMemory for LOLA OS state and context management.

Purpose: Manages state persistence, conversation history, and entity extraction for agents.
How: Uses JSON for persistence, litellm for entity extraction, and in-memory storage for conversation history.
Why: Ensures robust state management and context retention, supporting developer sovereignty and reliability.
Full Path: lola-os/python/lola/core/memory.py
"""

class ConversationMemory(BaseModel):
    """ConversationMemory: Manages dialog history for agents."""

    history: tp.List[tp.Dict[str, str]] = []

    def add_message(self, role: str, content: str) -> None:
        """
        Adds a message to the conversation history.

        Args:
            role: Message role (e.g., "user", "assistant").
            content: Message content.

        Does Not: Persist to storage—use StateManager.
        """
        self.history.append({"role": role, "content": content})

    def get_context(self) -> str:
        """
        Returns formatted conversation context for LLM.

        Returns:
            String of concatenated messages.

        Does Not: Include entities—use EntityMemory.
        """
        return "\n".join(f"{msg['role']}: {msg['content']}" for msg in self.history)

class EntityMemory(BaseModel):
    """EntityMemory: Extracts and stores key entities from conversations."""

    entities: tp.Dict[str, str] = {}

    def extract_entities(self, text: str, model: str) -> None:
        """
        Extracts entities using LLM via litellm.

        Args:
            text: Input text to analyze.
            model: LLM model string (e.g., "openai/gpt-4o").

        Does Not: Persist entities—use StateManager.
        """
        prompt = f"Extract key entities (e.g., people, places, facts) from: {text}"
        response = litellm.completion(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200
        )
        # Inline: Simplified entity parsing for Phase 1
        entities = response.choices[0].message.content.split(", ")
        for entity in entities:
            self.entities[entity] = entity

class StateManager:
    """StateManager: Handles state persistence and checkpointing (Item 15)."""

    def __init__(self, storage_path: str = "state.json"):
        """
        Initialize StateManager with a storage path.

        Args:
            storage_path: Path for JSON state storage.

        Does Not: Handle cross-session loading—expanded in Phase 2.
        """
        self.storage_path = Path(storage_path)
        self.conversation_memory = ConversationMemory()
        self.entity_memory = EntityMemory()

    def save_state(self, state: State) -> None:
        """
        Saves state to JSON file.

        Args:
            state: State instance to persist.

        Does Not: Handle database storage—Phase 2.
        """
        with self.storage_path.open("w") as f:
            json.dump(state.dict(), f)

    def load_state(self) -> State:
        """
        Loads state from JSON file.

        Returns:
            State instance or new State if file doesn't exist.

        Does Not: Handle errors—caller must validate.
        """
        if self.storage_path.exists():
            with self.storage_path.open("r") as f:
                return State(**json.load(f))
        return State()

    def update_conversation(self, role: str, content: str) -> None:
        """
        Updates conversation memory.

        Args:
            role: Message role.
            content: Message content.
        """
        self.conversation_memory.add_message(role, content)

    def extract_entities(self, text: str, model: str) -> None:
        """
        Extracts entities using EntityMemory.

        Args:
            text: Input text.
            model: LLM model string.
        """
        self.entity_memory.extract_entities(text, model)

__all__ = ["StateManager", "ConversationMemory", "EntityMemory"]