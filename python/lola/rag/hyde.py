# Standard imports
import typing as tp

# Local
from lola.agnostic.unified import UnifiedModelManager

"""
File: Defines the HyDEGenerator class for LOLA OS TMVP 1 Phase 2.

Purpose: Improves RAG retrieval with hypothetical documents.
How: Uses LLM to generate hypothetical answers for embedding.
Why: Enhances search accuracy, per Developer Sovereignty tenet.
Full Path: lola-os/python/lola/rag/hyde.py
Future Optimization: Migrate to Rust for fast generation (post-TMVP 1).
"""

class HyDEGenerator:
    """HyDEGenerator: Generates hypothetical documents for RAG. Does NOT persist data—use StateManager."""

    def __init__(self, model_manager: UnifiedModelManager):
        """
        Initialize with model manager.

        Args:
            model_manager: UnifiedModelManager instance.
        """
        self.model_manager = model_manager

    async def generate(self, query: str) -> str:
        """
        Generates a hypothetical document.

        Args:
            query: Input query.

        Returns:
            Hypothetical document string.

        Does Not: Embed—use multimodal.py.
        """
        prompt = f"Generate a hypothetical ideal document for query: {query}"
        return await self.model_manager.call(prompt)

__all__ = ["HyDEGenerator"]