# Standard imports
import typing as tp

# Third-party
from llama_index.core import SimpleDirectoryReader

# Local
from lola.rag.multimodal import MultiModalRetriever

"""
File: Defines the DynamicDataConnector class for LOLA OS TMVP 1 Phase 2.

Purpose: Syncs data from sources to RAG.
How: Uses LlamaIndex to load and index data.
Why: Enables dynamic knowledge bases, per Developer Sovereignty tenet.
Full Path: lola-os/python/lola/rag/connector.py
Future Optimization: Migrate to Rust for high-throughput syncing (post-TMVP 1).
"""

class DynamicDataConnector:
    """DynamicDataConnector: Syncs data to RAG. Does NOT persist data—use StateManager."""

    def __init__(self, retriever: MultiModalRetriever):
        """
        Initialize with retriever.

        Args:
            retriever: MultiModalRetriever instance.
        """
        self.retriever = retriever

    async def sync(self, source: str) -> None:
        """
        Syncs data from source.

        Args:
            source: Data source (e.g., file path).

        Does Not: Handle web sources—expand in TMVP 2.
        """
        if source.endswith('.txt') or source.endswith('.pdf'):
            reader = SimpleDirectoryReader(input_files=[source])
            data = reader.load_data()
            await self.retriever.index_data(data)

__all__ = ["DynamicDataConnector"]