# Standard imports
import typing as tp
from typing import List, Optional, Dict, Any
import os
from pathlib import Path

# Third-party
try:
    from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext
    from llama_index.core.node_parser import SentenceSplitter
    from llama_index.core.query_engine import RetrieverQueryEngine
    from llama_index.core.retrievers import VectorIndexRetriever
    from llama_index.embeddings.openai import OpenAIEmbedding
    from llama_index.llms.openai import OpenAI
except ImportError:
    raise ImportError("LlamaIndex not installed. Run 'poetry add llama-index-core llama-index-embeddings-openai'")

# Local
from lola.utils.config import get_config
from lola.utils.logging import logger
from lola.rag.multimodal import MultiModalRetriever  # Interconnection to Phase 2
from lola.libs.vector_dbs.adapter import VectorDBAdapter  # For flexible storage

"""
File: Wraps LlamaIndex retrievers for LOLA RAG module.
Purpose: Provides adapters for LlamaIndex's data ingestion, indexing, and retrieval in LOLA's RAG system.
How: Builds indexes from documents/directories, supports flexible VectorDB backends, 
     and provides query interfaces compatible with LOLA's MultiModalRetriever.
Why: Leverages LlamaIndex's superior data connectors and indexing while maintaining 
     LOLA's abstraction layer and VectorDB flexibility (Pinecone, FAISS, etc.).
Full Path: lola-os/python/lola/libs/llamaindex/retrievers.py
"""

class LlamaIndexRetrieverAdapter:
    """LlamaIndexRetrieverAdapter: Comprehensive wrapper for LlamaIndex retrievers.
    Does NOT handle VectorDB storage directly—uses libs/vector_dbs/ for flexibility."""

    DEFAULT_CHUNK_SIZE = 1024
    DEFAULT_CHUNK_OVERLAP = 200

    def __init__(self):
        """
        Initializes the LlamaIndex adapter with config and VectorDB integration.
        Does Not: Create indexes—lazy loading per use case.
        """
        config = get_config()
        self.enabled = config.get("use_llamaindex", True)  # Default enabled for RAG
        self.sentry_dsn = config.get("sentry_dsn", None)
        
        if not self.enabled:
            logger.warning("LlamaIndex disabled in config; RAG will use alternatives.")
            return

        # Load embedding model config
        embedding_config = config.get("embeddings", {})
        self.embedding_model = embedding_config.get("model", "text-embedding-ada-002")
        self.embedding_api_key = embedding_config.get("api_key")
        
        # Initialize VectorDB adapter for storage flexibility
        vector_config = config.get("vector_db", {})
        self.vector_db_type = vector_config.get("type", "pinecone")
        self.vector_db_adapter = self._get_vector_db_adapter(vector_config)
        
        logger.info(f"LlamaIndex adapter initialized with VectorDB: {self.vector_db_type}")

    def _get_vector_db_adapter(self, config: Dict[str, Any]) -> VectorDBAdapter:
        """
        Creates VectorDB adapter based on config.
        Args:
            config: VectorDB configuration.
        Returns:
            Initialized VectorDBAdapter instance.
        """
        from lola.libs.vector_dbs.adapter import VectorDBAdapter, get_vector_db_adapter
        
        try:
            # Inline: Factory pattern for VectorDB selection
            adapter = get_vector_db_adapter(config)
            logger.info(f"VectorDB adapter created: {self.vector_db_type}")
            return adapter
        except Exception as exc:
            logger.error(f"Failed to create VectorDB adapter: {str(exc)}")
            if self.sentry_dsn:
                from sentry_sdk import capture_exception
                capture_exception(exc)
            # Fallback to in-memory for testing
            return VectorDBAdapter(config={"type": "memory"})

    def build_index_from_directory(
        self, 
        directory_path: str, 
        chunk_size: Optional[int] = None, 
        chunk_overlap: Optional[int] = None
    ) -> VectorStoreIndex:
        """
        Builds a LlamaIndex from a directory of documents.
        Args:
            directory_path: Path to directory containing documents (PDF, TXT, MD, etc.).
            chunk_size: Chunk size for text splitting (default: 1024).
            chunk_overlap: Chunk overlap for context preservation (default: 200).
        Returns:
            LlamaIndex VectorStoreIndex instance.
        Does Not: Persist index—use persist_index() for storage.
        """
        if not self.enabled:
            raise ValueError("LlamaIndex disabled in config.")

        try:
            # Validate directory exists
            if not os.path.exists(directory_path):
                raise FileNotFoundError(f"Directory not found: {directory_path}")

            # Configure chunking
            chunk_size = chunk_size or self.DEFAULT_CHUNK_SIZE
            chunk_overlap = chunk_overlap or self.DEFAULT_CHUNK_OVERLAP
            
            # Inline: Use SentenceSplitter for intelligent text chunking
            node_parser = SentenceSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )

            # Load documents
            logger.info(f"Loading documents from: {directory_path}")
            documents = SimpleDirectoryReader(
                input_dir=directory_path,
                required_exts=[".pdf", ".txt", ".md", ".docx"],  # Expand as needed
                file_extractor={
                    ".pdf": "llama_index.readers.file.PDFReader",
                    ".docx": "llama_index.readers.file.DocxReader"
                }
            ).load_data()
            
            if not documents:
                raise ValueError(f"No supported documents found in: {directory_path}")

            logger.info(f"Loaded {len(documents)} documents")

            # Configure embeddings
            embed_model = OpenAIEmbedding(
                model=self.embedding_model,
                api_key=self.embedding_api_key
            )

            # Create index with VectorDB storage
            storage_context = StorageContext.from_defaults(
                vector_store=self.vector_db_adapter.get_store()
            )
            
            # Inline: Use custom node parser for better chunking
            index = VectorStoreIndex.from_documents(
                documents,
                storage_context=storage_context,
                embed_model=embed_model,
                transformations=[node_parser]
            )
            
            logger.info(f"Index built successfully with {len(index.docstore.docs)} nodes")
            return index

        except Exception as exc:
            error_msg = f"LlamaIndex build_index_from_directory failed: {directory_path}"
            self._handle_error(exc, error_msg)
            raise

    def build_index_from_text(
        self, 
        text_content: str, 
        metadata: Optional[Dict[str, Any]] = None
    ) -> VectorStoreIndex:
        """
        Builds index from raw text content.
        Args:
            text_content: Raw text to index.
            metadata: Optional metadata for the document.
        Returns:
            LlamaIndex VectorStoreIndex.
        """
        if not self.enabled:
            raise ValueError("LlamaIndex disabled in config.")

        try:
            from llama_index.core import Document
            
            # Create document with metadata
            doc = Document(
                text=text_content,
                metadata=metadata or {}
            )
            
            # Configure embeddings
            embed_model = OpenAIEmbedding(
                model=self.embedding_model,
                api_key=self.embedding_api_key
            )
            
            # Create simple index
            index = VectorStoreIndex.from_documents(
                [doc],
                embed_model=embed_model
            )
            
            logger.info(f"Text index built with {len(text_content)} characters")
            return index

        except Exception as exc:
            error_msg = "LlamaIndex build_index_from_text failed"
            self._handle_error(exc, error_msg)
            raise

    def query_index(
        self, 
        index: VectorStoreIndex, 
        query: str, 
        top_k: int = 5, 
        similarity_threshold: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Queries the LlamaIndex with similarity search.
        Args:
            index: LlamaIndex to query.
            query: Search query string.
            top_k: Number of results to return.
            similarity_threshold: Minimum similarity score (0.0-1.0).
        Returns:
            List of results with text, metadata, and similarity scores.
        """
        if not self.enabled:
            raise ValueError("LlamaIndex disabled in config.")

        try:
            # Create retriever with similarity threshold
            retriever = VectorIndexRetriever(
                index=index,
                similarity_top_k=top_k
            )
            
            # Create query engine
            query_engine = RetrieverQueryEngine.from_args(retriever)
            
            # Execute query
            response = query_engine.query(query)
            
            # Format results for LOLA compatibility
            results = []
            for node in response.source_nodes:
                result = {
                    "text": node.text,
                    "metadata": node.metadata,
                    "score": node.score or 0.0,
                    "node_id": node.node_id,
                    "hash": node.hash
                }
                
                # Apply similarity threshold if specified
                if similarity_threshold and node.score < similarity_threshold:
                    continue
                    
                results.append(result)
            
            logger.debug(f"Retrieved {len(results)} nodes for query: {query[:50]}...")
            return results[:top_k]  # Ensure we don't exceed top_k

        except Exception as exc:
            error_msg = f"LlamaIndex query failed: {query[:50]}..."
            self._handle_error(exc, error_msg)
            raise

    def persist_index(self, index: VectorStoreIndex, persist_path: str) -> None:
        """
        Persists index to disk (for local storage).
        Args:
            index: Index to persist.
            persist_path: Directory path for storage.
        Does Not: Handle VectorDB persistence—that's handled by vector_db_adapter.
        """
        if not self.enabled:
            raise ValueError("LlamaIndex disabled in config.")

        try:
            # Create persist directory
            Path(persist_path).mkdir(parents=True, exist_ok=True)
            
            # Persist index (local storage only)
            if self.vector_db_type in ["faiss", "chroma"]:
                index.storage_context.persist(persist_dir=persist_path)
                logger.info(f"Index persisted to: {persist_path}")
            else:
                logger.warning(f"Skipping local persist for VectorDB type: {self.vector_db_type}")
                
        except Exception as exc:
            error_msg = f"LlamaIndex persist failed: {persist_path}"
            self._handle_error(exc, error_msg)
            raise

    def load_persisted_index(self, persist_path: str) -> Optional[VectorStoreIndex]:
        """
        Loads persisted index from disk.
        Args:
            persist_path: Directory path where index was persisted.
        Returns:
            Loaded VectorStoreIndex or None if failed.
        """
        if not self.enabled:
            raise ValueError("LlamaIndex disabled in config.")

        try:
            if self.vector_db_type in ["faiss", "chroma"]:
                storage_context = StorageContext.from_defaults(persist_dir=persist_path)
                index = VectorStoreIndex.load_from_disk(persist_path, storage_context)
                logger.info(f"Loaded persisted index from: {persist_path}")
                return index
            else:
                logger.warning(f"Cannot load persisted index for VectorDB type: {self.vector_db_type}")
                return None
                
        except Exception as exc:
            error_msg = f"LlamaIndex load_persisted_index failed: {persist_path}"
            self._handle_error(exc, error_msg)
            logger.warning(f"Failed to load index, returning None: {str(exc)}")
            return None

    def _handle_error(self, exc: Exception, context: str) -> None:
        """
        Centralized error handling with logging and Sentry.
        Args:
            exc: Exception to handle.
            context: Context string for logging.
        """
        logger.error(f"{context}: {str(exc)}")
        if self.sentry_dsn:
            from sentry_sdk import capture_exception
            capture_exception(exc)

    # Integration method for LOLA RAG module
    def integrate_with_lola_rag(self, rag_component: MultiModalRetriever) -> None:
        """
        Registers this adapter with LOLA's MultiModalRetriever.
        Args:
            rag_component: LOLA RAG component to integrate with.
        Does Not: Modify RAG component internals—uses callback registration.
        """
        if not self.enabled:
            logger.warning("Cannot integrate LlamaIndex with RAG - adapter disabled")
            return

        try:
            # Register retriever callback
            def llama_retrieve(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
                # This would use a pre-built index - implementation depends on RAG usage
                if not hasattr(self, '_active_index'):
                    logger.warning("No active index for LlamaIndex retrieval")
                    return []
                
                return self.query_index(self._active_index, query, top_k)
            
            rag_component.register_retriever("llamaindex", llama_retrieve)
            logger.info("LlamaIndex retriever registered with LOLA RAG")
            
        except Exception as exc:
            self._handle_error(exc, "Failed to integrate LlamaIndex with LOLA RAG")


# Convenience functions for easy integration
def create_llamaindex_from_directory(
    directory_path: str, 
    chunk_size: int = 1024, 
    persist: bool = False,
    persist_path: Optional[str] = None
) -> VectorStoreIndex:
    """Quick factory to create index from directory."""
    adapter = LlamaIndexRetrieverAdapter()
    index = adapter.build_index_from_directory(directory_path, chunk_size)
    
    if persist and persist_path:
        adapter.persist_index(index, persist_path)
    
    return index

def query_llamaindex_documents(
    directory_path: str, 
    query: str, 
    top_k: int = 5
) -> List[Dict[str, Any]]:
    """Convenience function: build and query in one call."""
    adapter = LlamaIndexRetrieverAdapter()
    index = adapter.build_index_from_directory(directory_path)
    adapter._active_index = index  # For integration
    return adapter.query_index(index, query, top_k)

__all__ = [
    "LlamaIndexRetrieverAdapter",
    "create_llamaindex_from_directory",
    "query_llamaindex_documents"
]