# Standard imports
import typing as tp

# Local
from .base import BaseTemplateAgent
from lola.core.state import State
from lola.core.memory import ConversationMemory
from lola.tools.base import BaseTool

"""
File: Defines the ConversationalAgent class for LOLA OS TMVP 1 Phase 2.

Purpose: Implements a conversational agent for interactive dialogs.
How: Uses ConversationMemory to maintain history and LLM for responses.
Why: Enables natural user interactions, per Developer Sovereignty tenet.
Full Path: lola-os/python/lola/agents/conversational.py
Future Optimization: Migrate to Rust for high-throughput conversations (post-TMVP 1).
"""

class ConversationalAgent(BaseTemplateAgent):
    """ConversationalAgent: Implements interactive conversation pattern. Does NOT persist state—use StateManager."""

    def __init__(self, tools: tp.List[BaseTool], model: str = "openai/gpt-4o"):
        """
        Initialize with tools and LLM model.

        Args:
            tools: List of BaseTool instances.
            model: LLM model string for litellm.
        """
        super().__init__(tools, model)
        self.conversation_memory = ConversationMemory()

    async def run(self, query: str) -> State:
        """
        Respond to a query in conversation.

        Args:
            query: User input string.
        Returns:
            State: Updated state with response.
        Does Not: Handle entity extraction—use memory/entity.py.
        """
        self.state.update({"query": query})
        # Inline: Add query to conversation memory
        self.conversation_memory.add_message("user", query)
        # Inline: Generate response with LLM
        prompt = f"Respond to conversation: {self.conversation_memory.get_context()}"
        response = await self._call_llm(prompt)
        self.conversation_memory.add_message("assistant", response)
        self.state.update({"output": response})
        return self.state

__all__ = ["ConversationalAgent"]