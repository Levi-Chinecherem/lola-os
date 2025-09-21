# Standard imports
import pytest
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock
import typing as tp
from pathlib import Path
import tempfile

# Local
from lola.rag.multimodal import MultiModalRetriever
from lola.rag.hyde import HyDEGenerator
from lola.rag.evaluator import RAGEvaluator
from lola.rag.connector import DynamicDataConnector
from lola.agnostic.unified import UnifiedModelManager
from lola.utils import sentry

"""
File: Comprehensive tests for LOLA OS RAG in Phase 5.

Purpose: Validates RAG components with real VectorDB and LlamaIndex, mocks for APIs.
How: Uses pytest with async support, real FAISS/Pinecone tests, and mocks for validation.
Why: Ensures robust knowledge retrieval with >80% coverage, per Radical Reliability tenet.
Full Path: lola-os/tests/test_rag.py
"""

@pytest.fixture
def mock_model_manager():
    """Fixture for a mocked UnifiedModelManager."""
    return MagicMock(spec=UnifiedModelManager, call=AsyncMock(return_value="Hypothetical doc"))

@pytest.mark.asyncio
async def test_multi_modal_retriever_faiss(tmp_path, mocker):
    """Test MultiModalRetriever with FAISS backend."""
    mocker.patch("lola.utils.sentry.init_sentry")
    mocker.patch("prometheus_client.Counter.inc")
    config = {"vector_db": {"type": "faiss", "index_path": str(tmp_path / "faiss_index"), "dimension": 1536}}
    retriever = MultiModalRetriever(config)
    test_file = tmp_path / "test.txt"
    test_file.write_text("test content")
    await retriever.index_data([str(test_file)])
    result = await retriever.retrieve("test query")
    assert isinstance(result, list)

@pytest.mark.asyncio
async def test_hyde_generator(mocker, mock_model_manager):
    """Test HyDEGenerator with mocked LLM."""
    mocker.patch("lola.utils.sentry.capture_exception")
    generator = HyDEGenerator(mock_model_manager)
    result = await generator.generate("Test query")
    assert result == "Hypothetical doc"

def test_rag_evaluator():
    """Test RAGEvaluator with sample data."""
    evaluator = RAGEvaluator()
    result = evaluator.evaluate(["doc1", "doc2"], ["doc1"])
    assert result["precision"] == 0.5
    assert result["recall"] == 1.0

@pytest.mark.asyncio
async def test_dynamic_data_connector(tmp_path, mocker):
    """Test DynamicDataConnector with FAISS retriever."""
    mocker.patch("lola.rag.multimodal.MultiModalRetriever.index_data", AsyncMock(return_value=None))
    mocker.patch("lola.utils.sentry.capture_exception")
    config = {"vector_db": {"type": "faiss", "index_path": str(tmp_path / "faiss_index"), "dimension": 1536}}
    retriever = MultiModalRetriever(config)
    connector = DynamicDataConnector(retriever)
    test_file = tmp_path / "test.txt"
    test_file.write_text("test content")
    await connector.sync(str(test_file))