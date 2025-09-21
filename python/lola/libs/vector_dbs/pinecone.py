# Standard imports
import typing as tp
import logging

# Third-party
from pinecone import Pinecone, ServerlessSpec
from tenacity import retry, stop_after_attempt, wait_exponential

# Local
from .adapter import VectorDBAdapter
from lola.utils import sentry

"""
File: Implements PineconeAdapter for LOLA OS TMVP 1 Phase 5.

Purpose: Provides Pinecone-specific vector DB operations.
How: Uses pinecone-client for indexing and querying.
Why: Enables cloud-based vector search, per Developer Sovereignty tenet.
Full Path: lola-os/python/lola/libs/vector_dbs/pinecone.py
"""

logger = logging.getLogger(__name__)

class PineconeAdapter(VectorDBAdapter):
    """PineconeAdapter: Implements vector DB operations using Pinecone."""

    def __init__(self, config: tp.Dict[str, tp.Any]):
        """
        Initialize Pinecone adapter.

        Args:
            config: Dict with 'api_key' and 'index_name'.
        """
        self.initialize(config)

    def initialize(self, config: tp.Dict[str, tp.Any]) -> None:
        """
        Initialize Pinecone with API key and index.

        Args:
            config: Dict with 'api_key' and 'index_name'.

        Does Not: Create embeddings—use rag/.
        """
        try:
            api_key = config.get("api_key")
            if not api_key:
                logger.error("Pinecone API key required")
                sentry.capture_message("Pinecone API key missing")
                raise ValueError("Pinecone API key required")
            self.pc = Pinecone(api_key=api_key)
            index_name = config.get("index_name", "lola-index")
            if index_name not in self.pc.list_indexes().names():
                self.pc.create_index(
                    name=index_name,
                    dimension=1536,
                    metric="cosine",
                    spec=ServerlessSpec(cloud="aws", region="us-west-2")
                )
                logger.info(f"Created Pinecone index: {index_name}")
            self.index = self.pc.Index(index_name)
        except Exception as e:
            logger.error(f"Pinecone initialization failed: {e}")
            sentry.capture_exception(e)
            raise

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def query(self, vector: tp.List[float], top_k: int) -> tp.List[tp.Any]:
        """
        Query Pinecone index.

        Args:
            vector: Query embedding.
            top_k: Number of results.

        Returns:
            List of matching vectors.
        """
        try:
            response = self.index.query(vector=vector, top_k=top_k, include_metadata=True)
            logger.info(f"Pinecone query returned {len(response.matches)} results")
            return response.matches
        except Exception as e:
            logger.error(f"Pinecone query failed: {e}")
            sentry.capture_exception(e)
            raise

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def index(self, vectors: tp.List[tp.Dict[str, tp.Any]]) -> None:
        """
        Index vectors in Pinecone.

        Args:
            vectors: List of dicts with id, vector, metadata.
        """
        try:
            self.index.upsert(vectors=[(v["id"], v["vector"], v.get("metadata", {})) for v in vectors])
            logger.info(f"Indexed {len(vectors)} vectors in Pinecone")
        except Exception as e:
            logger.error(f"Pinecone indexing failed: {e}")
            sentry.capture_exception(e)
            raise

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def delete(self, ids: tp.List[str]) -> None:
        """
        Delete vectors by IDs.

        Args:
            ids: List of vector IDs.
        """
        try:
            self.index.delete(ids=ids)
            logger.info(f"Deleted {len(ids)} vectors from Pinecone")
        except Exception as e:
            logger.error(f"Pinecone deletion failed: {e}")
            sentry.capture_exception(e)
            raise

__all__ = ["PineconeAdapter"]