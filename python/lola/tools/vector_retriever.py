# Standard imports
import typing as tp

# Third-party
from pinecone import Pinecone, ServerlessSpec

# Local
from .base import BaseTool

"""
File: Defines the VectorDBRetrieverTool class for LOLA OS TMVP 1 Phase 2.

Purpose: Enables agents to retrieve from vector databases.
How: Uses Pinecone for real vector retrieval.
Why: Supports knowledge retrieval, per Developer Sovereignty tenet.
Full Path: lola-os/python/lola/tools/vector_retriever.py
Future Optimization: Migrate to Rust for fast vector operations (post-TMVP 1).
"""

class VectorDBRetrieverTool(BaseTool):
    """VectorDBRetrieverTool: Retrieves from Pinecone vector DB. Does NOT persist data—use StateManager."""

    name: str = "vector_retriever"

    def __init__(self, api_key: str, index_name: str = "lola-index"):
        """
        Initialize with Pinecone API key and index name.

        Args:
            api_key: Pinecone API key.
            index_name: Pinecone index name.
        """
        self.pc = Pinecone(api_key=api_key)
        if index_name not in self.pc.list_indexes().names():
            self.pc.create_index(name=index_name, dimension=1536, metric='cosine', spec=ServerlessSpec(cloud='aws', region='us-west-2'))
        self.index = self.pc.Index(index_name)

    async def execute(self, input_data: tp.Any) -> tp.Any:
        """
        Retrieve from vector DB.

        Args:
            input_data: Dict with 'query' embedding (list of floats).

        Returns:
            List of matching vectors.

        Does Not: Generate embeddings—use llama-index in rag/.
        """
        if not isinstance(input_data, dict) or 'query' not in input_data:
            raise ValueError("Input data must be dict with 'query' embedding.")
        response = self.index.query(vector=input_data['query'], top_k=5)
        return response.matches

__all__ = ["VectorDBRetrieverTool"]