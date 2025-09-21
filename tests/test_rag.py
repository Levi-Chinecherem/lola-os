# Standard imports
import pytest
import asyncio
from unittest.mock import patch, MagicMock

# Local
from lola.rag.multimodal import MultiModalRetriever
from lola.rag.hyde import HyDEGenerator
from lola.rag.evaluator import RAGEvaluator
from lola.rag.connector import DynamicDataConnector
from lola.agnostic.unified import UnifiedModelManager

"""
File: Comprehensive tests for LOLA OS RAG in Phase 2.

Purpose: Validates RAG components with real Pinecone and LlamaIndex, mocks for APIs.
How: Uses pytest with async support, patch for LLM calls, and test data for validation.
Why: Ensures robust knowledge retrieval with >80% coverage, per Radical Reliability tenet.
Full Path: lola-os/tests/test_rag.py
"""

@pytest.mark.asyncio
async def test_multi_modal_retriever(tmp_path, mocker):
    """Test MultiModalRetriever indexing and retrieval with mocked Pinecone."""
    mocker.patch('pinecone.Pinecone', return_value=MagicMock(Index=MagicMock(query=MagicMock(return_value={"matches": [{"text": "match"}]}))))
    retriever = MultiModalRetriever(pinecone_api_key="test_key")
    test_file = tmp_path / "test.txt"
    test_file.write_text("test content")
    await retriever.index_data([str(test_file)])
    result = await retriever.retrieve("test query")
    assert isinstance(result, list)
    assert len(result) > 0

@pytest.mark.asyncio
async def test_hyde_generator(mocker):
    """Test HyDEGenerator with mocked LLM."""
    mocker.patch('lola.agnostic.unified.UnifiedModelManager.call', return_value="Hypothetical doc")
    manager = UnifiedModelManager("test/model")
    generator = HyDEGenerator(manager)
    result = await generator.generate("Test query")
    assert result == "Hypothetical doc"

def test_rag_evaluator():
    """Test RAGEvaluator with sample data."""
    evaluator = RAGEvaluator()
    result = evaluator.evaluate(["doc1", "doc2"], ["doc1"])
    assert result["precision"] == 0.5
    assert result["recall"] == 1.0

@pytest.mark.asyncio
async def test_dynamic_data_connector(mocker):
    """Test DynamicDataConnector with mocked retriever."""
    mocker.patch('lola.rag.multimodal.MultiModalRetriever.index_data', return_value=None)
    retriever = MagicMock(spec=MultiModalRetriever)
    connector = DynamicDataConnector(retriever)
    test_file = "test.txt"
    await connector.sync(test_file)

# Run tests if executed directly
if __name__ == "__main__":
    pytest.main()