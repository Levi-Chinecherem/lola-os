# Standard imports
import pytest
import tempfile
import os
from pathlib import Path
import typing as tp

# Local
from lola.libs.llamaindex.retrievers import LlamaIndexRetrieverAdapter
from lola.utils.config import get_config

"""
Test file for LlamaIndex retriever adapter.
Purpose: Ensures LlamaIndex integration works with flexible VectorDB backends 
         and proper error handling.
Full Path: lola-os/tests/test_libs_llamaindex_retrievers.py
"""

@pytest.fixture
def mock_config():
    """Mock config with LlamaIndex enabled."""
    with patch('lola.utils.config.get_config') as mock:
        mock.return_value = {
            "use_llamaindex": True,
            "embeddings": {
                "model": "text-embedding-ada-002",
                "api_key": "fake_embedding_key"
            },
            "vector_db": {"type": "memory"},  # Use memory for testing
            "sentry_dsn": "test_dsn"
        }
        yield mock

@pytest.fixture
def adapter(mock_config):
    """Fixture for LlamaIndexRetrieverAdapter."""
    return LlamaIndexRetrieverAdapter()

@pytest.fixture
def sample_text_content():
    """Sample text for testing."""
    return """
    LOLA OS is an open-source framework for building EVM-native AI agents.
    It provides a Python-first interface with seamless integration 
    for blockchain interactions and LLM providers.
    The framework emphasizes developer sovereignty and radical reliability.
    """

def test_adapter_initialization(adapter):
    """Test adapter initializes correctly."""
    assert adapter.enabled is True
    assert adapter.embedding_model == "text-embedding-ada-002"
    assert hasattr(adapter, 'vector_db_adapter')
    assert adapter.vector_db_type == "memory"

def test_adapter_disabled_raises(adapter):
    """Test adapter raises when disabled."""
    adapter.enabled = False
    with pytest.raises(ValueError, match="LlamaIndex disabled in config"):
        adapter.build_index_from_text("test")

@patch('sentry_sdk.capture_exception')
def test_vector_db_creation_error_handling(mock_sentry, adapter):
    """Test VectorDB adapter creation handles errors."""
    with patch('lola.libs.llamaindex.retrievers.VectorDBAdapter') as mock_adapter:
        mock_adapter.side_effect = Exception("VectorDB init failed")
        
        # Force reinitialization to trigger error
        adapter.vector_db_adapter = None
        with pytest.raises(Exception):
            adapter._get_vector_db_adapter({"type": "broken"})
        
        mock_sentry.assert_called_once()

def test_build_index_from_text_success(adapter, tmpdir):
    """Test building index from text content."""
    text_content = sample_text_content()
    
    index = adapter.build_index_from_text(text_content)
    
    assert index is not None
    assert hasattr(index, 'as_query_engine')
    assert len(index.docstore.docs) == 1

def test_query_index_success(adapter):
    """Test querying an index returns formatted results."""
    # Create index first
    text_content = sample_text_content()
    index = adapter.build_index_from_text(text_content)
    
    # Query the index
    results = adapter.query_index(index, "LOLA OS framework", top_k=3)
    
    assert isinstance(results, list)
    assert len(results) > 0
    assert "text" in results[0]
    assert "metadata" in results[0]
    assert "score" in results[0]

def test_query_with_similarity_threshold(adapter):
    """Test similarity threshold filtering works."""
    text_content = sample_text_content()
    index = adapter.build_index_from_text(text_content)
    
    # Query with high threshold (should return fewer results)
    results = adapter.query_index(
        index, 
        "blockchain", 
        top_k=5, 
        similarity_threshold=0.8
    )
    
    # With high threshold, might get fewer or no results
    if results:
        assert all(r["score"] >= 0.8 for r in results)
    else:
        # This is acceptable - high threshold might filter everything
        pass

def test_build_index_from_directory(tmpdir, adapter):
    """Test building index from directory with sample files."""
    # Create sample files
    sample_dir = tmpdir.mkdir("documents")
    
    # Create text file
    text_file = sample_dir.join("lola_os.txt")
    text_file.write(sample_text_content(), encoding="utf-8")
    
    # Create another file
    doc_file = sample_dir.join("features.md")
    doc_file.write("""
    # LOLA OS Features
    
    ## EVM Integration
    - Read-only contract calls
    - Event listening
    - Oracle data fetching
    
    ## AI Capabilities
    - Multi-LLM support via LiteLLM
    - Advanced RAG pipelines
    - Agent orchestration
    """, encoding="utf-8")
    
    # Build index from directory
    index = adapter.build_index_from_directory(str(sample_dir))
    
    assert index is not None
    assert len(index.docstore.docs) >= 2  # Should find both files

def test_persist_and_load_index(tmpdir, adapter):
    """Test index persistence and loading (for local VectorDBs)."""
    # Only test persistence for local VectorDBs
    if adapter.vector_db_type not in ["faiss", "chroma", "memory"]:
        pytest.skip(f"Skipping persist test for VectorDB: {adapter.vector_db_type}")
    
    # Create index
    text_content = sample_text_content()
    index = adapter.build_index_from_text(text_content)
    
    # Persist
    persist_path = str(tmpdir.mkdir("persist"))
    adapter.persist_index(index, persist_path)
    
    # Verify files were created
    assert os.path.exists(persist_path)
    assert len(os.listdir(persist_path)) > 0
    
    # Load back
    loaded_index = adapter.load_persisted_index(persist_path)
    
    assert loaded_index is not None
    assert len(loaded_index.docstore.docs) == 1

def test_directory_not_found_error(adapter, tmpdir):
    """Test error handling for non-existent directory."""
    non_existent_dir = str(tmpdir.join("nonexistent"))
    
    with pytest.raises(FileNotFoundError, match="Directory not found"):
        adapter.build_index_from_directory(non_existent_dir)

def test_empty_directory_error(adapter, tmpdir):
    """Test error handling for empty directory."""
    empty_dir = tmpdir.mkdir("empty")
    
    with pytest.raises(ValueError, match="No supported documents found"):
        adapter.build_index_from_directory(str(empty_dir))

def test_integration_with_lola_rag(mocker, adapter):
    """Test integration with LOLA's MultiModalRetriever."""
    # Mock the RAG component
    mock_rag = mocker.Mock(spec=MultiModalRetriever)
    mock_rag.register_retriever = mocker.Mock()
    
    # Test integration
    adapter.integrate_with_lola_rag(mock_rag)
    
    # Verify registration was called
    mock_rag.register_retriever.assert_called_once_with(
        "llamaindex", 
        mocker.ANY  # The retriever function
    )

def test_convenience_functions():
    """Test convenience functions work."""
    from lola.libs.llamaindex.retrievers import create_llamaindex_from_directory
    
    # This would require actual files, so just test it doesn't crash
    with pytest.raises(FileNotFoundError):
        create_llamaindex_from_directory("/nonexistent/path")

# Performance test (optional)
@pytest.mark.performance
def test_index_build_performance(adapter, tmpdir):
    """Test index building performance with larger dataset."""
    # Create larger test data
    large_dir = tmpdir.mkdir("large_docs")
    for i in range(10):  # 10 documents
        file_path = large_dir.join(f"doc_{i}.txt")
        content = sample_text_content() * (i + 1)  # Varying lengths
        file_path.write(content, encoding="utf-8")
    
    import time
    start_time = time.time()
    index = adapter.build_index_from_directory(str(large_dir))
    build_time = time.time() - start_time
    
    assert index is not None
    assert len(index.docstore.docs) == 10
    print(f"Index build time: {build_time:.2f}s")

# Coverage marker
@pytest.mark.coverage
def test_coverage_marker(adapter):
    """Marker for coverage reporting."""
    pass

# Run with: pytest tests/test_libs_llamaindex_retrievers.py -v --cov=lola/libs/llamaindex --cov-report=html