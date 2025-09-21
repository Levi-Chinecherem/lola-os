# Standard imports
import pytest
import asyncio
import typing as tp
from unittest.mock import AsyncMock, patch
from pathlib import Path
import tempfile

# Third-party
import pandas as pd

# Local
from lola.tools import (
    BaseTool,
    WebSearchTool,
    DataAnalysisTool,
    CodeInterpreterTool,
    HumanInputTool,
    FileIOTool,
    APIClientTool,
    VectorDBRetrieverTool,
    ContractCallTool,
    EventListenerTool,
    OracleFetchTool
)
from lola.chains.contract import Contract
from lola.chains.connection import ChainConnection
from lola.chains.oracles import Oracles

"""
File: Tests for LOLA OS tools in Phase 2.

Purpose: Validates tool functionality with >80% coverage, including async execution and error cases.
How: Uses pytest with async support, mocks for external services (e.g., DuckDuckGo, Pinecone, web3.py).
Why: Ensures robust tool implementations, per Radical Reliability tenet.
Full Path: lola-os/tests/test_tools.py
"""

@pytest.fixture
def mock_state():
    """Fixture for an initial state."""
    return {"query": "", "history": [], "metadata": {}}

@pytest.mark.asyncio
async def test_web_search_tool(mocker):
    """Test WebSearchTool with mocked DuckDuckGo."""
    mocker.patch("duckduckgo_search.AsyncDDGS.text", AsyncMock(return_value=[
        {"title": "Test", "body": "Snippet", "href": "http://test.com"}
    ]))
    tool = WebSearchTool()
    result = await tool.execute("test query")
    assert len(result) == 1
    assert result[0]["title"] == "Test"

@pytest.mark.asyncio
async def test_data_analysis_tool():
    """Test DataAnalysisTool with pandas."""
    tool = DataAnalysisTool()
    input_data = {"operation": "mean", "data": [{"a": 1, "b": 2}, {"a": 3, "b": 4}]}
    result = await tool.execute(input_data)
    assert result == {"a": 2.0, "b": 3.0}

    input_data = {"operation": "describe", "data": [{"a": 1, "b": 2}, {"a": 3, "b": 4}]}
    result = await tool.execute(input_data)
    assert "mean" in result["a"]

@pytest.mark.asyncio
async def test_code_interpreter_tool():
    """Test CodeInterpreterTool execution."""
    tool = CodeInterpreterTool()
    result = await tool.execute("print('Hello')")
    assert result == "Hello"

    result = await tool.execute("invalid syntax")
    assert "Error: " in result

@pytest.mark.asyncio
async def test_human_input_tool(mocker):
    """Test HumanInputTool with mocked input."""
    mocker.patch("builtins.input", return_value="User input")
    tool = HumanInputTool()
    result = await tool.execute("Enter text")
    assert result == "User input"

@pytest.mark.asyncio
async def test_file_io_tool():
    """Test FileIOTool read/write operations."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.txt"
        tool = FileIOTool()

        # Test write
        result = await tool.execute({"operation": "write", "path": str(path), "content": "Test content"})
        assert result == "File written successfully"
        assert path.read_text() == "Test content"

        # Test read
        result = await tool.execute({"operation": "read", "path": str(path)})
        assert result == "Test content"

@pytest.mark.asyncio
async def test_api_client_tool(mocker):
    """Test APIClientTool with mocked requests."""
    mocker.patch("requests.get", return_value=type("Response", (), {"status_code": 200, "json": lambda: {"result": "OK"}, "text": '{"result": "OK"}'}))
    tool = APIClientTool()
    result = await tool.execute({"url": "http://test.com", "method": "GET"})
    assert result == {"result": "OK"}

@pytest.mark.asyncio
async def test_vector_retriever_tool(mocker):
    """Test VectorDBRetrieverTool with mocked Pinecone."""
    mocker.patch("pinecone.Pinecone.__init__", return_value=None)
    mocker.patch("pinecone.Pinecone.list_indexes", return_value=type("Indexes", (), {"names": lambda: ["lola-index"]})())
    mocker.patch("pinecone.Pinecone.Index", return_value=type("Index", (), {"query": lambda *args, **kwargs: type("Response", (), {"matches": [{"id": "1", "score": 0.9}]})}))
    tool = VectorDBRetrieverTool