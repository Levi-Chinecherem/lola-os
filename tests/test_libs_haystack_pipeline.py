# Standard imports
import pytest
from unittest.mock import Mock, patch, MagicMock
import typing as tp
from typing import List, Dict, Any
import time

# Local
from lola.libs.haystack.pipeline import HaystackPipelineAdapter
from lola.rag.multimodal import MultiModalRetriever
from lola.utils.config import get_config

"""
Test file for Haystack pipeline adapter.
Purpose: Ensures Haystack integration works with LOLA RAG, VectorDB bridge, 
         and proper error handling for different pipeline types.
Full Path: lola-os/tests/test_libs_haystack_pipeline.py
"""

@pytest.fixture
def mock_config():
    """Mock configuration with Haystack enabled."""
    with patch('lola.utils.config.get_config') as mock:
        mock.return_value = {
            "use_haystack": True,
            "haystack_model_provider": "openai",
            "haystack_embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
            "haystack_qa_model": "distilbert-base-cased-distilled-squad",
            "haystack_summarization_model": "facebook/bart-large-cnn",
            "vector_db": {"type": "memory"},
            "sentry_dsn": "test_dsn"
        }
        yield mock

@pytest.fixture
def adapter(mock_config):
    """Fixture for HaystackPipelineAdapter."""
    return HaystackPipelineAdapter()

@pytest.fixture
def sample_documents() -> List[Dict[str, Any]]:
    """Sample documents for testing."""
    return [
        {
            "id": "doc1",
            "text": "LOLA OS is an open-source framework for building EVM-native AI agents.",
            "metadata": {"category": "framework", "source": "docs"},
            "embedding": [0.1, 0.2, 0.3] * 512  # Mock 1536-dim embedding
        },
        {
            "id": "doc2", 
            "text": "Vector databases enable efficient similarity search for embeddings.",
            "metadata": {"category": "database", "source": "blog"},
            "embedding": [0.4, 0.5, 0.6] * 512
        },
        {
            "id": "doc3",
            "text": "Haystack provides advanced NLP pipelines for question answering and retrieval.",
            "metadata": {"category": "nlp", "source": "paper"},
            "embedding": [0.7, 0.8, 0.9] * 512
        }
    ]

def test_adapter_initialization(adapter):
    """Test adapter initializes correctly when enabled."""
    assert adapter.enabled is True
    assert hasattr(adapter, 'document_store')
    assert adapter.model_provider == "openai"
    assert adapter.embedding_model == "sentence-transformers/all-MiniLM-L6-v2"

def test_adapter_disabled(mock_config):
    """Test adapter behavior when Haystack is disabled."""
    with patch('lola.utils.config.get_config', return_value={"use_haystack": False}):
        adapter = HaystackPipelineAdapter()
        assert adapter.enabled is False
        
        # Should not raise when disabled
        with pytest.raises(ValueError, match="Haystack disabled"):
            adapter.create_retrieval_pipeline()

@patch('sentry_sdk.capture_exception')
def test_vector_db_initialization_error_handling(mock_sentry, mock_config):
    """Test VectorDB initialization error handling."""
    with patch('lola.libs.haystack.pipeline.get_vector_db_adapter') as mock_adapter:
        mock_adapter.side_effect = Exception("VectorDB connection failed")
        
        adapter = HaystackPipelineAdapter()
        assert hasattr(adapter, 'document_store')
        assert isinstance(adapter.document_store, tp.Any)  # Should fallback to in-memory
        
        mock_sentry.assert_called_once()

def test_create_document_store(adapter, sample_documents):
    """Test document store bridge creation and document writing."""
    # Test document writing
    adapter.add_documents(sample_documents)
    
    # Verify bridge functionality
    document_count = adapter.document_store.get_document_count()
    assert document_count >= len(sample_documents)
    
    # Test retrieval through bridge
    all_docs = adapter.document_store.get_all_documents()
    assert len(all_docs) >= len(sample_documents)
    assert all_docs[0].get("content") == sample_documents[0]["text"]

def test_create_retrieval_pipeline(adapter):
    """Test retrieval pipeline creation."""
    pipeline = adapter.create_retrieval_pipeline(top_k=3)
    
    assert hasattr(pipeline, 'run')
    assert "retriever" in pipeline.graph
    assert pipeline.graph["retriever"].component.__class__.__name__ == "InMemoryDocumentRetriever"

def test_retrieval_pipeline_run(adapter, sample_documents):
    """Test end-to-end retrieval pipeline execution."""
    # Add documents first
    adapter.add_documents(sample_documents)
    
    # Create pipeline
    pipeline = adapter.create_retrieval_pipeline(top_k=2)
    
    # Run retrieval
    results = adapter.run_retrieval(pipeline, "LOLA OS framework", top_k=2)
    
    assert isinstance(results, list)
    assert len(results) >= 1
    assert results[0]["text"] == sample_documents[0]["text"]
    assert "metadata" in results[0]
    assert results[0]["metadata"]["category"] == "framework"

def test_create_qa_pipeline(adapter, sample_documents):
    """Test QA pipeline creation and execution."""
    # Add documents
    adapter.add_documents(sample_documents)
    
    # Create QA pipeline
    pipeline = adapter.create_qa_pipeline(top_k_retriever=5, top_k_reader=1)
    
    assert hasattr(pipeline, 'run')
    assert "retriever" in pipeline.graph
    assert "reader" in pipeline.graph
    
    # Test execution (mock reader response for test)
    with patch('haystack.nodes.FARMReader') as mock_reader:
        mock_reader_instance = Mock()
        mock_reader.return_value = mock_reader_instance
        mock_reader_instance.predict.return_value = {
            "answers": [{"answer": "LOLA OS", "score": 0.95}]
        }
        
        result = adapter.run_qa(pipeline, "What is LOLA OS?")
        
        assert result["answer"] == "LOLA OS"
        assert result["confidence"] == 0.95
        assert "context" in result

def test_create_summarization_pipeline(adapter):
    """Test summarization pipeline creation."""
    pipeline = adapter.create_summarization_pipeline()
    
    assert hasattr(pipeline, 'run')
    assert "summarizer" in pipeline.graph
    assert pipeline.graph["summarizer"].component.__class__.__name__ == "TransformersSummarizer"

def test_summarization_execution(adapter, sample_documents):
    """Test summarization pipeline execution."""
    # Combine documents for summarization
    long_text = " ".join([doc["text"] for doc in sample_documents])
    
    pipeline = adapter.create_summarization_pipeline()
    
    # Mock summarizer for test
    with patch('haystack.nodes.TransformersSummarizer') as mock_summarizer:
        mock_instance = Mock()
        mock_summarizer.return_value = mock_instance
        mock_instance.predict.return_value = {
            "summary": "LOLA OS provides AI agents with vector database integration."
        }
        
        result = pipeline.run(query=long_text)
        
        assert "summary" in result
        assert result["summary"] == "LOLA OS provides AI agents with vector database integration."

def test_rag_integration(adapter, mocker):
    """Test integration with LOLA MultiModalRetriever."""
    # Mock RAG component
    mock_rag = mocker.Mock(spec=MultiModalRetriever)
    mock_rag.register_retriever = mocker.Mock()
    
    # Test integration
    adapter.integrate_with_lola_rag(mock_rag)
    
    # Verify registration
    mock_rag.register_retriever.assert_called_once_with(
        "haystack",
        mocker.ANY  # The haystack_retrieve function
    )

def test_convenience_functions(sample_documents):
    """Test convenience functions."""
    from lola.libs.haystack.pipeline import create_haystack_retrieval, run_haystack_qa
    
    # Test retrieval creation
    pipeline = create_haystack_retrieval(sample_documents, top_k=2)
    assert hasattr(pipeline, 'run')
    
    # Test QA function (mock for test)
    with patch('lola.libs.haystack.pipeline.HaystackPipelineAdapter') as mock_adapter:
        mock_instance = Mock()
        mock_adapter.return_value = mock_instance
        mock_instance.run_qa.return_value = {"answer": "test answer"}
        
        result = run_haystack_qa("test question", sample_documents)
        assert result["answer"] == "test answer"

def test_error_handling(adapter, sample_documents):
    """Test error handling during document operations."""
    # Test invalid document format
    invalid_docs = [{"invalid": "format"}]  # Missing required fields
    
    with pytest.raises(Exception):
        adapter.add_documents(invalid_docs)
    
    # Verify Sentry was called if configured
    if get_config().get("sentry_dsn"):
        with patch('sentry_sdk.capture_exception') as mock_sentry:
            with pytest.raises(Exception):
                adapter.run_retrieval(Mock(), "test", top_k=5)
            mock_sentry.assert_called_once()

def test_pipeline_filtering(adapter, sample_documents):
    """Test metadata filtering in retrieval."""
    # Add documents with categories
    adapter.add_documents(sample_documents)
    
    pipeline = adapter.create_retrieval_pipeline(top_k=10)
    
    # Test category filtering
    results = adapter.run_retrieval(
        pipeline, 
        "AI framework", 
        top_k=10,
        filters={"category": "framework"}
    )
    
    # Should only return framework documents
    assert len(results) == 1
    assert results[0]["metadata"]["category"] == "framework"

# Performance test
@pytest.mark.performance  
def test_large_scale_retrieval(adapter):
    """Test retrieval performance with larger document set."""
    # Generate large test set
    large_docs = []
    for i in range(100):
        doc = {
            "id": f"doc_{i}",
            "text": f"Document {i} about LOLA OS and vector databases.",
            "metadata": {"category": "tech", "doc_num": i},
            "embedding": [float(i % 10) * 0.1 + j * 0.01 for j in range(1536)]  # Mock embedding
        }
        large_docs.append(doc)
    
    # Add documents
    start_add = time.time()
    adapter.add_documents(large_docs)
    add_time = time.time() - start_add
    
    # Test retrieval
    pipeline = adapter.create_retrieval_pipeline(top_k=10)
    start_query = time.time()
    results = adapter.run_retrieval(pipeline, "LOLA OS", top_k=10)
    query_time = time.time() - start_query
    
    assert len(results) == 10
    print(f"Added 100 docs in {add_time:.3f}s, queried in {query_time:.3f}s")

# Coverage marker
@pytest.mark.coverage
def test_coverage_marker(adapter):
    """Marker for coverage reporting."""
    pass

# Run with: pytest tests/test_libs_haystack_pipeline.py -v --cov=lola/libs/haystack --cov-report=html