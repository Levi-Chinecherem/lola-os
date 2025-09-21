"""
File: Initializes the vector_dbs module for LOLA OS TMVP 1 Phase 5.

Purpose: Exports vector database adapters for flexible retrieval.
How: Defines package-level exports for developer imports.
Why: Centralizes access to VectorDB implementations, per Developer Sovereignty.
Full Path: lola-os/python/lola/libs/vector_dbs/__init__.py
"""

from .adapter import VectorDBAdapter
from .pinecone import PineconeAdapter
from .faiss import FaissAdapter
from .chroma import ChromaAdapter
from .postgres import PostgresAdapter

__all__ = [
    "VectorDBAdapter",
    "PineconeAdapter",
    "FaissAdapter",
    "ChromaAdapter",
    "PostgresAdapter",
]