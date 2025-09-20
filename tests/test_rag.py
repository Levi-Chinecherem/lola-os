# Standard imports
import pytest
import typing as tp

# Local
from lola.rag import MultiModalRetriever, HyDEGenerator, RAGEvaluator, DynamicDataConnector
from lola.agnostic.unified import UnifiedModelManager

"""
File: Tests for RAG module in LOLA OS TMVP 1 Phase 2.

Purpose: Verifies RAG component initialization and functionality.
How: Uses pytest to test RAG classes.
Why: Ensures robust retrieval, per Developer Sovereignty.
Full Path: lola-os/tests/test_rag.py
"""
@pytest.mark.asyncio
async def test_rag_functionality():
    """Test RAG component functionality."""
    model_manager = UnifiedModelManager()
    retriever = MultiModalRetriever(model_manager)
    hyde = HyDEGenerator(model_manager)
    evaluator = RAGEvaluator()
    connector = DynamicDataConnector()

    assert isinstance(await retriever.retrieve("test"), dict)
    assert isinstance(await hyde.generate("test"), dict)
    assert isinstance(evaluator.evaluate("test", {}), dict)
    assert isinstance(connector.sync("test"), dict)