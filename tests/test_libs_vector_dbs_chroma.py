# Standard imports
import pytest
import tempfile
import os
import numpy as np
from typing import List, Dict, Any
import uuid
from unittest.mock import patch

# Local
from lola.libs.vector_dbs.chroma import ChromaVectorDBAdapter, create_chroma_adapter
from lola.utils.config import get_config

"""
Test file for Chroma VectorDB adapter.
Purpose: Ensures Chroma integration works correctly with persistence, 
         metadata filtering, and error handling.
Full Path: lola-os/tests/test_libs_vector_dbs_chroma.py
"""

@pytest.fixture
def mock_config():
    """Mock configuration for Chroma testing."""
    with patch('lola.utils.config.get_config') as mock:
        mock.return_value = {
            "sentry_dsn": "test_dsn",
            "chroma": {
                "allow_reset": True
            }
        }
        yield mock

@pytest.fixture
def temp_chroma_dir(tmpdir):
    """Temporary directory for Chroma persistence."""
    chroma_dir = tmpdir.mkdir("chroma_test")
    return str(chroma_dir)

@pytest.fixture
def chroma_adapter(temp_chroma_dir, mock_config):
    """Fixture for ChromaVectorDBAdapter."""
    config = {
        "type": "chroma",
        "persist_directory": temp_chroma_dir,
        "collection_name": "test_collection",
        "embedding_dim": 3,  # Small dimension for testing
        "allow_reset": True
    }
    adapter = ChromaVectorDBAdapter(config)
    adapter.connect()
    yield adapter
    adapter.disconnect()

@pytest.fixture
def sample_embeddings() -> List[List[float]]:
    """Sample 3D embeddings for testing."""
    return [
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
        [0.7, 0.8, 0.9],
        [0.2, 0.1, 0.4]
    ]

@pytest.fixture
def sample_texts() -> List[str]:
    """Sample texts for testing."""
    return [
        "LOLA OS is an AI agent framework",
        "Vector databases store embeddings efficiently", 
        "ChromaDB provides local vector storage",
        "FAISS is optimized for similarity search"
    ]

@pytest.fixture
def sample_metadatas() -> List[Dict[str, Any]]:
    """Sample metadata for testing."""
    return [
        {"category": "framework", "source": "docs"},
        {"category": "database", "source": "blog"},
        {"category": "database", "source": "docs"},
        {"category": "search", "source": "paper"}
    ]

def test_adapter_initialization(chroma_adapter, temp_chroma_dir):
    """Test Chroma adapter initializes correctly."""
    assert chroma_adapter.db_type == "chroma"
    assert chroma_adapter.persist_directory == temp_chroma_dir
    assert chroma_adapter.collection_name == "test_collection"
    assert chroma_adapter._embedding_dim == 3
    assert chroma_adapter.settings.persist_directory == temp_chroma_dir

def test_connection_success(chroma_adapter):
    """Test successful connection to Chroma."""
    assert chroma_adapter._connected is True
    assert chroma_adapter._client is not None
    assert chroma_adapter._collection is not None
    assert chroma_adapter._collection.name == "test_collection"

def test_indexing_success(chroma_adapter, sample_embeddings, sample_texts, sample_metadatas):
    """Test indexing embeddings with metadata."""
    # Index data
    chroma_adapter.index(sample_embeddings, sample_texts, sample_metadatas)
    
    # Verify count
    stats = chroma_adapter.get_stats()
    assert stats["count"] == 4
    assert stats["dimensions"] == 3
    assert stats["type"] == "chroma"

def test_query_success(chroma_adapter, sample_embeddings, sample_texts, sample_metadatas):
    """Test querying returns relevant results."""
    # First index some data
    chroma_adapter.index(sample_embeddings, sample_texts, sample_metadatas)
    
    # Query with first embedding
    query_embedding = sample_embeddings[0]  # Should match itself
    results = chroma_adapter.query(query_embedding, top_k=3)
    
    assert len(results) == 3
    assert results[0]["id"] is not None
    assert results[0]["distance"] >= 0.0
    assert results[0]["text"] == "LOLA OS is an AI agent framework"
    assert "metadata" in results[0]
    assert results[0]["metadata"]["category"] == "framework"

def test_query_with_metadata_filtering(chroma_adapter, sample_embeddings, sample_texts, sample_metadatas):
    """Test metadata-based filtering in queries."""
    # Index data
    chroma_adapter.index(sample_embeddings, sample_texts, sample_metadatas)
    
    # Query with metadata filter (only database category)
    query_embedding = sample_embeddings[0]
    results = chroma_adapter.query(
        query_embedding, 
        top_k=5,
        where={"category": "database"}
    )
    
    # Should only return database-related results
    assert len(results) == 2
    for result in results:
        assert result["metadata"]["category"] == "database"

def test_query_with_text_filtering(chroma_adapter, sample_embeddings, sample_texts, sample_metadatas):
    """Test text content filtering."""
    # Index data
    chroma_adapter.index(sample_embeddings, sample_texts, sample_metadatas)
    
    # Query filtering for documents containing "vector"
    query_embedding = sample_embeddings[0]
    results = chroma_adapter.query(
        query_embedding,
        top_k=5,
        where_document={"$contains": "vector"}
    )
    
    # Should find vector-related documents
    assert len(results) >= 1
    for result in results:
        assert "vector" in result["text"].lower()

def test_delete_success(chroma_adapter, sample_embeddings, sample_texts, sample_metadatas):
    """Test deleting specific vectors."""
    # Index data
    ids = [str(uuid.uuid4()) for _ in sample_embeddings]
    chroma_adapter.index(sample_embeddings, sample_texts, sample_metadatas, ids=ids)
    
    # Delete first two
    before_count = chroma_adapter.get_stats()["count"]
    assert before_count == 4
    
    chroma_adapter.delete(ids[:2])
    
    # Verify deletion
    after_count = chroma_adapter.get_stats()["count"]
    assert after_count == 2
    
    # Query should not return deleted items
    query_embedding = sample_embeddings[0]
    results = chroma_adapter.query(query_embedding, top_k=5)
    deleted_ids = set(ids[:2])
    remaining_ids = set(ids[2:])
    
    for result in results:
        assert result["id"] not in deleted_ids
        assert result["id"] in remaining_ids

def test_persistence_across_sessions(temp_chroma_dir):
    """Test data persists across adapter instances."""
    # First session - index data
    config = {
        "type": "chroma",
        "persist_directory": temp_chroma_dir,
        "collection_name": "persist_test",
        "embedding_dim": 3
    }
    
    adapter1 = ChromaVectorDBAdapter(config)
    adapter1.connect()
    
    sample_emb = [[0.1, 0.2, 0.3]]
    sample_text = ["Persistent test document"]
    sample_meta = [{"test": "data"}]
    
    adapter1.index(sample_emb, sample_text, sample_meta)
    assert adapter1.get_stats()["count"] == 1
    
    adapter1.disconnect()
    
    # Second session - load data
    adapter2 = ChromaVectorDBAdapter(config)
    adapter2.connect()
    
    stats = adapter2.get_stats()
    assert stats["count"] == 1
    assert stats["is_persistent"] is True
    
    # Query to verify data
    results = adapter2.query(sample_emb[0], top_k=1)
    assert len(results) == 1
    assert results[0]["text"] == "Persistent test document"
    
    adapter2.disconnect()

def test_reset_collection(chroma_adapter):
    """Test collection reset functionality."""
    # First add some data
    sample_emb = [[0.1, 0.2, 0.3]]
    sample_text = ["Reset test"]
    sample_meta = [{"test": "reset"}]
    
    chroma_adapter.index(sample_emb, sample_text, sample_meta)
    assert chroma_adapter.get_stats()["count"] == 1
    
    # Reset collection
    chroma_adapter.reset_collection(confirm=True)
    
    # Verify reset
    stats = chroma_adapter.get_stats()
    assert stats["count"] == 0

def test_reset_without_confirmation(chroma_adapter):
    """Test reset requires confirmation."""
    with pytest.raises(ValueError, match="Must pass confirm=True"):
        chroma_adapter.reset_collection(confirm=False)

def test_multiple_collections(chroma_adapter, temp_chroma_dir):
    """Test multiple collections in same instance."""
    # Create second adapter with different collection
    config2 = chroma_adapter.config.copy()
    config2["collection_name"] = "second_collection"
    
    adapter2 = ChromaVectorDBAdapter(config2)
    adapter2.connect()
    
    # Index in both collections
    sample_emb = [[0.1, 0.2, 0.3]]
    sample_text = ["First collection"]
    sample_meta = [{"collection": "first"}]
    
    sample_emb2 = [[0.4, 0.5, 0.6]]
    sample_text2 = ["Second collection"] 
    sample_meta2 = [{"collection": "second"}]
    
    chroma_adapter.index(sample_emb, sample_text, sample_meta)
    adapter2.index(sample_emb2, sample_text2, sample_meta2)
    
    # Verify separate counts
    assert chroma_adapter.get_stats()["count"] == 1
    assert adapter2.get_stats()["count"] == 1
    
    # Verify collection names
    assert chroma_adapter.get_collection_names() == ["test_collection", "second_collection"]

def test_error_handling(chroma_adapter, sample_embeddings, sample_texts, sample_metadatas):
    """Test error handling during indexing."""
    # Test dimension mismatch
    bad_embeddings = [[0.1, 0.2]]  # Wrong dimension (2 instead of 3)
    
    with pytest.raises(ValueError, match="All embeddings must have 3 dimensions"):
        chroma_adapter.index(bad_embeddings, sample_texts[:1], sample_metadatas[:1])

def test_factory_function():
    """Test convenience factory function."""
    adapter = create_chroma_adapter(
        persist_directory="./test_chroma",
        embedding_dim=128,
        collection_name="factory_test"
    )
    
    assert isinstance(adapter, ChromaVectorDBAdapter)
    assert adapter._embedding_dim == 128
    assert adapter.collection_name == "factory_test"

# Performance test
@pytest.mark.performance
def test_large_scale_indexing(temp_chroma_dir):
    """Test performance with larger dataset."""
    config = {
        "type": "chroma",
        "persist_directory": temp_chroma_dir,
        "collection_name": "perf_test",
        "embedding_dim": 1536
    }
    
    adapter = ChromaVectorDBAdapter(config)
    adapter.connect()
    
    # Generate test data
    n_vectors = 1000
    dim = 1536
    embeddings = np.random.random((n_vectors, dim)).astype(np.float32).tolist()
    texts = [f"Document {i}" for i in range(n_vectors)]
    metadatas = [{"doc_id": i} for i in range(n_vectors)]
    
    import time
    start_time = time.time()
    adapter.index(embeddings, texts, metadatas)
    index_time = time.time() - start_time
    
    # Verify indexing
    stats = adapter.get_stats()
    assert stats["count"] == n_vectors
    
    # Test query performance
    query_embedding = embeddings[0]
    start_query = time.time()
    results = adapter.query(query_embedding, top_k=10)
    query_time = time.time() - start_query
    
    print(f"Indexed {n_vectors} vectors in {index_time:.2f}s")
    print(f"Query time: {query_time:.4f}s")
    
    adapter.disconnect()

# Coverage marker
@pytest.mark.coverage
def test_coverage_marker(chroma_adapter):
    """Marker for coverage reporting."""
    pass

# Run with: pytest tests/test_libs_vector_dbs_chroma.py -v --cov=lola/libs/vector_dbs/chroma --cov-report=html