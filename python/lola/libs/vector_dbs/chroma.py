# Standard imports
import typing as tp
import numpy as np
from typing import List, Dict, Any, Optional
import uuid
from pathlib import Path

# Third-party
try:
    import chromadb
    from chromadb.config import Settings
except ImportError:
    raise ImportError("Chroma not installed. Run 'poetry add chromadb'")

# Local
from lola.libs.vector_dbs.adapter import VectorDBAdapter
from lola.utils.config import get_config
from lola.utils.logging import logger
from sentry_sdk import capture_exception

"""
File: Chroma VectorDB implementation for LOLA OS.
Purpose: Provides persistent, local-first vector storage using ChromaDB with 
         support for both file-based and in-memory persistence.
How: Uses Chroma Client with configurable persistence directory; supports 
     collections, metadata filtering, and automatic schema management.
Why: Excellent for local development, containerized deployments, and scenarios 
     requiring SQLite-based persistence without external services.
Full Path: lola-os/python/lola/libs/vector_dbs/chroma.py
"""

class ChromaVectorDBAdapter(VectorDBAdapter):
    """ChromaVectorDBAdapter: Local-first vector storage using ChromaDB.
    Does NOT require external services; supports both file and in-memory modes."""

    DEFAULT_COLLECTION_NAME = "lola_vectors"

    def __init__(self, config: Dict[str, Any]):
        """
        Initializes Chroma adapter.
        Args:
            config: Configuration with 'persist_directory' and 'collection_name'.
        """
        super().__init__(config)
        self.persist_directory = config.get("persist_directory", "./chroma_db")
        self.collection_name = config.get("collection_name", self.DEFAULT_COLLECTION_NAME)
        self.allow_reset = config.get("allow_reset", True)
        self._client = None
        self._collection = None
        self._connected = False
        
        # Configure Chroma settings
        self.settings = Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory=str(Path(self.persist_directory).absolute()),
            anonymized_telemetry=False  # Respect privacy
        )
        
        logger.info(f"Chroma adapter initialized: {self.persist_directory}")

    def connect(self) -> None:
        """Connects to Chroma and ensures collection exists."""
        try:
            # Initialize Chroma client
            self._client = chromadb.Client(self.settings)
            
            # Get or create collection
            try:
                self._collection = self._client.get_collection(self.collection_name)
                logger.debug(f"Loaded existing Chroma collection: {self.collection_name}")
            except Exception:
                # Collection doesn't exist, create it
                self._collection = self._client.create_collection(
                    name=self.collection_name,
                    metadata={
                        "lola_embedding_dim": self._embedding_dim,
                        "hnsw_space": "cosine"  # Use cosine similarity
                    }
                )
                logger.info(f"Created new Chroma collection: {self.collection_name}")
            
            self._connected = True
            
            # Test connection with stats
            stats = self.get_stats()
            logger.info(f"Chroma connected: {stats['count']} vectors in {stats['dimensions']}D space")
            
        except Exception as exc:
            self._handle_error(exc, "Chroma connection")
            raise

    def disconnect(self) -> None:
        """Closes Chroma client and flushes changes."""
        try:
            if self._client and self._connected:
                # Persist any pending changes
                self._client.persist()
                logger.debug(f"Chroma persisted changes to: {self.persist_directory}")
            
            self._client = None
            self._collection = None
            self._connected = False
            logger.debug("Chroma disconnected")
            
        except Exception as exc:
            logger.warning(f"Chroma disconnect warning: {str(exc)}")

    def index(self, embeddings: List[List[float]], texts: List[str], 
              metadatas: List[Dict[str, Any]], ids: Optional[List[str]] = None) -> None:
        """Indexes embeddings in Chroma collection."""
        self.ensure_connected()
        
        try:
            # Validate inputs
            n_vectors = len(embeddings)
            if n_vectors != len(texts) or n_vectors != len(metadatas):
                raise ValueError("All input lists must have equal lengths")
            
            if any(len(emb) != self._embedding_dim for emb in embeddings):
                raise ValueError(f"All embeddings must have {self._embedding_dim} dimensions")

            # Generate IDs if not provided
            if ids is None:
                ids = [str(uuid.uuid4()) for _ in range(n_vectors)]

            # Prepare metadata with full text (Chroma supports longer text than Pinecone)
            documents = texts  # Full text storage
            formatted_metadatas = []
            for i, (text, metadata) in enumerate(zip(texts, metadatas)):
                # Store full text length in metadata for filtering
                full_metadata = {
                    **metadata,
                    "text_length": len(text),
                    "document_id": ids[i],
                    "indexed_at": str(tp.datetime.now().isoformat())
                }
                formatted_metadatas.append(full_metadata)

            # Convert embeddings to numpy array (Chroma expects this)
            embeddings_array = np.array(embeddings, dtype=np.float32)

            # Add to collection in batches for better performance
            batch_size = 100
            for i in range(0, n_vectors, batch_size):
                batch_end = min(i + batch_size, n_vectors)
                batch_ids = ids[i:batch_end]
                batch_embeddings = embeddings_array[i:batch_end]
                batch_documents = documents[i:batch_end]
                batch_metadatas = formatted_metadatas[i:batch_end]
                
                self._collection.add(
                    embeddings=batch_embeddings.tolist(),
                    documents=batch_documents,
                    metadatas=batch_metadatas,
                    ids=batch_ids
                )
                
                logger.debug(f"Added batch {i//batch_size + 1}: {len(batch_ids)} vectors")

            logger.info(f"Successfully indexed {n_vectors} vectors in Chroma")
            
        except Exception as exc:
            self._handle_error(exc, "Chroma indexing")
            raise

    def query(self, embedding: List[float], top_k: int = 5, 
              include_metadata: bool = True, 
              where: Optional[Dict[str, Any]] = None,
              where_document: Optional[Dict[str, str]] = None) -> List[Dict[str, Any]]:
        """
        Queries Chroma collection with optional metadata filtering.
        Args:
            embedding: Query embedding vector.
            top_k: Number of results.
            include_metadata: Include metadata in results.
            where: Metadata filter (e.g., {"category": "tech"}).
            where_document: Text content filter.
        Returns:
            List of results with id, distance, document, and metadata.
        """
        self.ensure_connected()
        
        try:
            if self._collection.count() == 0:
                logger.debug("Chroma query: empty collection")
                return []

            # Convert embedding to numpy
            query_embedding = np.array([embedding], dtype=np.float32).tolist()[0]

            # Execute query with optional filters
            results = self._collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                include=["distances", "documents", "metadatas", "ids"],
                where=where,
                where_document=where_document
            )

            # Format results for LOLA compatibility
            formatted_results = []
            if results["ids"]:
                for i in range(len(results["ids"][0])):
                    result = {
                        "id": results["ids"][0][i],
                        "distance": float(results["distances"][0][i]),
                        "text": results["documents"][0][i] or "",
                        "embedding_dim": self._embedding_dim
                    }
                    
                    if include_metadata and results["metadatas"][0][i]:
                        result["metadata"] = results["metadatas"][0][i].copy()
                    
                    formatted_results.append(result)

            logger.debug(f"Chroma query returned {len(formatted_results)} results")
            return formatted_results

        except Exception as exc:
            self._handle_error(exc, "Chroma query")
            raise

    def delete(self, ids: List[str]) -> None:
        """Deletes vectors by IDs from Chroma collection."""
        self.ensure_connected()
        
        try:
            if not ids:
                logger.warning("Chroma delete: empty ID list")
                return

            before_count = self._collection.count()
            self._collection.delete(ids=ids)
            after_count = self._collection.count()
            
            deleted_count = before_count - after_count
            logger.info(f"Chroma deleted {deleted_count} vectors (remaining: {after_count})")
            
        except Exception as exc:
            self._handle_error(exc, "Chroma delete")
            raise

    def get_stats(self) -> Dict[str, Any]:
        """Returns Chroma collection statistics."""
        self.ensure_connected()
        try:
            count = self._collection.count()
            return {
                "type": "chroma",
                "count": count,
                "dimensions": self._embedding_dim,
                "collection_name": self.collection_name,
                "persist_directory": self.persist_directory,
                "is_persistent": Path(self.persist_directory).exists()
            }
        except Exception as exc:
            self._handle_error(exc, "Chroma stats")
            return {"type": "chroma", "count": 0, "dimensions": self._embedding_dim}

    def reset_collection(self, confirm: bool = False) -> None:
        """
        Resets the entire collection (dangerous operation).
        Args:
            confirm: Must be True to proceed.
        """
        if not self.allow_reset:
            raise PermissionError("Collection reset disabled in configuration")
        
        if not confirm:
            raise ValueError("Must pass confirm=True to reset collection")
        
        self.ensure_connected()
        try:
            self._client.delete_collection(self.collection_name)
            self.connect()  # Reconnect to create fresh collection
            logger.warning(f"Chroma collection '{self.collection_name}' reset")
        except Exception as exc:
            self._handle_error(exc, "Chroma collection reset")
            raise

    def get_collection_names(self) -> List[str]:
        """Returns all collection names in the Chroma instance."""
        if not self._connected or not self._client:
            return []
        
        try:
            return self._client.list_collections()
        except Exception as exc:
            logger.warning(f"Failed to list Chroma collections: {str(exc)}")
            return []

    def _handle_error(self, exc: Exception, context: str) -> None:
        """Error handling for Chroma operations."""
        full_context = f"Chroma[{self.collection_name}] {context}"
        logger.error(f"{full_context}: {str(exc)}")
        config = get_config()
        if config.get("sentry_dsn"):
            capture_exception(exc)


# Convenience factory
def create_chroma_adapter(
    persist_directory: str = "./chroma_db",
    collection_name: str = "lola_vectors",
    embedding_dim: int = 1536,
    allow_reset: bool = True
) -> ChromaVectorDBAdapter:
    """Creates Chroma adapter with sensible defaults."""
    config = {
        "type": "chroma",
        "persist_directory": persist_directory,
        "collection_name": collection_name,
        "embedding_dim": embedding_dim,
        "allow_reset": allow_reset
    }
    return ChromaVectorDBAdapter(config)

__all__ = [
    "ChromaVectorDBAdapter",
    "create_chroma_adapter"
]