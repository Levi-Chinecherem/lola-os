# Standard imports
import typing as tp
from typing import List, Dict, Any, Optional
import pickle
import numpy as np
from pathlib import Path

# Third-party
try:
    import faiss
    import hnswlib  # Alternative for large-scale similarity search
except ImportError:
    raise ImportError("FAISS not installed. Run 'poetry add faiss-cpu'")

# Sentry error tracking
try:
    from sentry_sdk import capture_exception
except ImportError:
    def capture_exception(exc):
        pass

# Local
from lola.libs.vector_dbs.adapter import VectorDBAdapter, get_vector_db_adapter
from lola.utils.config import get_config
from lola.utils.logging import logger

"""
File: FAISS VectorDB implementation for LOLA OS.
Purpose: Provides local, file-based vector storage using Facebook AI Similarity Search (FAISS).
How: Uses FAISS IndexFlatIP for cosine similarity; supports file persistence; 
     optimized for CPU with optional GPU support detection.
Why: Ideal for offline development, edge deployments, and scenarios requiring 
     no external API dependencies or data privacy.
Full Path: lola-os/python/lola/libs/vector_dbs/faiss.py
"""

class FAISSVectorDBAdapter(VectorDBAdapter):
    """FAISSVectorDBAdapter: Local vector storage using FAISS library.
    Does NOT require internet connectivity; optimized for CPU with GPU detection."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initializes FAISS adapter.
        Args:
            config: Configuration with 'index_path' for persistence.
        """
        super().__init__(config)
        self.index_path = config.get("index_path", "./faiss_index")
        self.use_gpu = config.get("use_gpu", False)
        self._index = None
        self._index_type = config.get("index_type", "flat")  # flat, ivf, hnsw
        
        # Detect GPU availability if requested
        if self.use_gpu:
            try:
                import faiss.contrib.gpu  # Test GPU support
                self._gpu_resource = faiss.StandardGpuResources()
                logger.info("FAISS GPU support detected and enabled")
            except ImportError:
                logger.warning("FAISS GPU support requested but not available, falling back to CPU")
                self.use_gpu = False
        
        self._connected = False
        logger.info(f"FAISS adapter initialized: {self.index_path}, GPU: {self.use_gpu}")

    def connect(self) -> None:
        """Loads or creates FAISS index."""
        try:
            index_path = Path(self.index_path)
            
            if index_path.exists():
                # Load existing index
                self._index = faiss.read_index(str(index_path))
                logger.info(f"Loaded FAISS index from: {self.index_path}")
            else:
                # Create new index
                if self._index_type == "flat":
                    self._index = faiss.IndexFlatIP(self._embedding_dim)
                elif self._index_type == "ivf":
                    nlist = self.config.get("nlist", 100)
                    quantizer = faiss.IndexFlatIP(self._embedding_dim)
                    self._index = faiss.IndexIVFFlat(quantizer, self._embedding_dim, nlist)
                    self._index.train(np.random.random((1000, self._embedding_dim)).astype('float32'))
                else:
                    raise ValueError(f"Unsupported FAISS index type: {self._index_type}")
                
                logger.info(f"Created new {self._index_type} FAISS index with dim {self._embedding_dim}")
            
            # Move to GPU if available
            if self.use_gpu and not isinstance(self._index, faiss.IndexFlatIP):
                self._index = faiss.index_cpu_to_gpu(self._gpu_resource, 0, self._index)
            
            self._connected = True
            stats = self.get_stats()
            logger.info(f"FAISS connected: {stats['count']} vectors")
            
        except Exception as exc:
            logger.error(f"FAISS connection failed: {str(exc)}")
            if get_config().get("sentry_dsn"):
                capture_exception(exc)
            raise

    def disconnect(self) -> None:
        """Saves index to disk and cleans up."""
        try:
            if self._index is not None and self._connected:
                # Save index
                Path(self.index_path).parent.mkdir(parents=True, exist_ok=True)
                faiss.write_index(self._index, str(self.index_path))
                logger.debug(f"FAISS index saved to: {self.index_path}")
            
            self._index = None
            self._connected = False
            
        except Exception as exc:
            logger.warning(f"FAISS disconnect warning: {str(exc)}")

    def index(self, embeddings: List[List[float]], texts: List[str], 
              metadatas: List[Dict[str, Any]], ids: Optional[List[str]] = None) -> None:
        """Indexes embeddings using FAISS."""
        self.ensure_connected()
        
        try:
            # Validate inputs
            if len(embeddings) != len(texts) or len(texts) != len(metadatas):
                raise ValueError("All input lists must have equal lengths")
            
            n_vectors = len(embeddings)
            if ids is None:
                ids = self._generate_ids(n_vectors, prefix="faiss")
            
            # Convert to numpy array
            vectors_array = np.array(embeddings, dtype=np.float32)
            
            # Normalize vectors for cosine similarity (IndexFlatIP expects normalized)
            faiss.normalize_L2(vectors_array)
            
            # Add to index
            self._index.add(vectors_array)
            
            # Store metadata separately (FAISS doesn't store metadata natively)
            metadata_path = Path(self.index_path).with_suffix('.metadata')
            metadata_storage = {
                'ids': ids,
                'texts': texts,
                'metadatas': metadatas,
                'embedding_dim': self._embedding_dim
            }
            
            # Load existing metadata if exists
            if metadata_path.exists():
                with open(metadata_path, 'rb') as f:
                    existing = pickle.load(f)
                existing['ids'].extend(ids)
                existing['texts'].extend(texts)
                existing['metadatas'].extend(metadatas)
                metadata_storage = existing
            
            # Save updated metadata
            with open(metadata_path, 'wb') as f:
                pickle.dump(metadata_storage, f)
            
            logger.info(f"Indexed {n_vectors} vectors in FAISS (total: {self.get_stats()['count']})")
            
        except Exception as exc:
            self._handle_error(exc, "FAISS indexing")
            raise

    def query(self, embedding: List[float], top_k: int = 5, 
              include_metadata: bool = True) -> List[Dict[str, Any]]:
        """Queries FAISS index for similar vectors."""
        self.ensure_connected()
        
        try:
            if self._index.ntotal == 0:
                logger.debug("FAISS query: empty index")
                return []

            # Prepare query vector
            query_array = np.array([embedding], dtype=np.float32)
            faiss.normalize_L2(query_array)
            
            # Search (IndexFlatIP returns similarities, convert to distances)
            distances, indices = self._index.search(query_array, top_k)
            distances = 1.0 - distances[0]  # Convert cosine similarity to distance
            
            # Load metadata
            metadata_path = Path(self.index_path).with_suffix('.metadata')
            if not metadata_path.exists():
                logger.warning("FAISS metadata not found")
                return [{"id": f"vec_{i}", "distance": float(d), "text": ""} 
                       for i, d in enumerate(distances)]
            
            with open(metadata_path, 'rb') as f:
                metadata_storage = pickle.load(f)
            
            ids = metadata_storage['ids']
            texts = metadata_storage['texts']
            metadatas = metadata_storage['metadatas']
            
            results = []
            for i, idx in enumerate(indices[0]):
                if idx == -1:  # FAISS returns -1 for out-of-bounds
                    continue
                
                result = {
                    "id": ids[idx],
                    "distance": float(distances[i]),
                    "text": texts[idx][:500] + "..." if len(texts[idx]) > 500 else texts[idx]
                }
                
                if include_metadata:
                    result["metadata"] = metadatas[idx].copy()
                
                results.append(result)
            
            return results[:top_k]
            
        except Exception as exc:
            self._handle_error(exc, "FAISS query")
            raise

    def delete(self, ids: List[str]) -> None:
        """FAISS doesn't support selective delete; rebuilds index."""
        self.ensure_connected()
        
        try:
            # Load current data
            metadata_path = Path(self.index_path).with_suffix('.metadata')
            if not metadata_path.exists():
                logger.warning("No metadata to delete from")
                return
            
            with open(metadata_path, 'rb') as f:
                metadata_storage = pickle.load(f)
            
            # Filter out IDs to delete
            ids_to_keep = [i for i, id in enumerate(metadata_storage['ids']) if id not in ids]
            keep_count = len(ids_to_keep)
            
            if keep_count == 0:
                # Clear everything
                self._index.reset()
                metadata_storage = {
                    'ids': [], 'texts': [], 'metadatas': [],
                    'embedding_dim': self._embedding_dim
                }
            else:
                # Rebuild index with remaining vectors
                remaining_vectors = [self._vectors[i] for i in ids_to_keep]  # Assumes _vectors stored
                vectors_array = np.array(remaining_vectors, dtype=np.float32)
                faiss.normalize_L2(vectors_array)
                
                self._index.reset()
                self._index.add(vectors_array)
                
                # Update metadata
                metadata_storage = {
                    'ids': [metadata_storage['ids'][i] for i in ids_to_keep],
                    'texts': [metadata_storage['texts'][i] for i in ids_to_keep],
                    'metadatas': [metadata_storage['metadatas'][i] for i in ids_to_keep],
                    'embedding_dim': self._embedding_dim
                }
            
            # Save updated metadata
            with open(metadata_path, 'wb') as f:
                pickle.dump(metadata_storage, f)
            
            deleted_count = len(ids)
            logger.info(f"FAISS delete: removed {deleted_count} vectors (remaining: {keep_count})")
            
        except Exception as exc:
            self._handle_error(exc, "FAISS delete")
            raise

    def get_stats(self) -> Dict[str, Any]:
        """Returns FAISS statistics."""
        self.ensure_connected()
        try:
            return {
                "type": "faiss",
                "count": self._index.ntotal,
                "dimensions": self._embedding_dim,
                "index_type": self._index_type,
                "is_trained": hasattr(self._index, 'is_trained') and self._index.is_trained,
                "gpu_enabled": self.use_gpu
            }
        except Exception as exc:
            self._handle_error(exc, "FAISS stats")
            return {"type": "faiss", "count": 0, "dimensions": self._embedding_dim}

    def _handle_error(self, exc: Exception, context: str) -> None:
        """Error handling for FAISS operations."""
        full_context = f"FAISS[{self._index_type}] {context}"
        logger.error(f"{full_context}: {str(exc)}")
        config = get_config()
        if config.get("sentry_dsn"):
            capture_exception(exc)


# Quick factory for common use
def create_faiss_index(
    embedding_dim: int = 1536,
    index_path: str = "./faiss_index",
    index_type: str = "flat",
    use_gpu: bool = False
) -> FAISSVectorDBAdapter:
    """Creates a FAISS adapter with common defaults."""
    config = {
        "type": "faiss",
        "embedding_dim": embedding_dim,
        "index_path": index_path,
        "index_type": index_type,
        "use_gpu": use_gpu
    }
    return FAISSVectorDBAdapter(config)

__all__ = [
    "FAISSVectorDBAdapter",
    "create_faiss_index"
]