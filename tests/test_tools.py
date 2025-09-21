# Standard imports
import pytest
import asyncio
from unittest.mock import patch, AsyncMock, MagicMock
import pandas as pd
from pathlib import Path
import requests
import typing as tp

# Third-party
import pinecone

# Local
from lola.tools.base import BaseTool
from lola.tools.web_search import WebSearchTool
from lola.tools.data_analysis import DataAnalysisTool
from lola.tools.code_interpreter import CodeInterpreterTool
from lola.tools.human_input import HumanInputTool
from lola.tools.file_io import FileIOTool
from lola.tools.api_client import APIClientTool
from lola.tools.vector_retriever import VectorDBRetrieverTool
from lola.tools.onchain.contract_call import ContractCallTool
from lola.tools.onchain.event_listener import EventListenerTool
from lola.tools.onchain.oracle_fetch import OracleFetchTool
from lola.chains.connection import ChainConnection

"""
File: Comprehensive tests for LOLA OS tools in Phase 2.

Purpose: Validates tool initialization, execution, and interconnections with mocks for external APIs.
How: Uses pytest with async support, patch for requests/web3/pinecone, and test data for validation.
Why: Ensures robust tool performance with >80% coverage, per Radical Reliability tenet.
Full Path: lola-os/tests/test_tools.py
"""

@pytest.mark.asyncio
async def test_web_search_tool(mocker):
    """Test WebSearchTool execution with mocked requests."""
    mocker.patch('requests.get', return_value=MagicMock(status_code=200, json=lambda: {"RelatedTopics": [{"Title": "Test", "Abstract": "Snippet", "FirstURL": "url"}]}))
    tool = WebSearchTool()
    result = await tool.execute("test query")
    assert isinstance(result, list)
    assert len(result) > 0
    assert "title" in result[0]

@pytest.mark.asyncio
async def test_web_search_tool_failure(mocker):
    """Test WebSearchTool failure handling with mocked requests."""
    mocker.patch('requests.get', side_effect=requests.RequestException("API failure"))
    tool = WebSearchTool()
    result = await tool.execute("test query")
    assert result == []  # Assuming WebSearchTool returns empty list on failure

@pytest.mark.asyncio
async def test_data_analysis_tool():
    """Test DataAnalysisTool with sample data."""
    tool = DataAnalysisTool()
    input_data = {"operation": "mean", "data": [{"value": 1}, {"value": 2}, {"value": 3}]}
    result = await tool.execute(input_data)
    assert isinstance(result, dict)
    assert result["value"] == 2.0

@pytest.mark.asyncio
async def test_data_analysis_tool_invalid_input():
    """Test DataAnalysisTool with invalid input."""
    tool = DataAnalysisTool()
    input_data = {"operation": "invalid", "data": [{"value": 1}]}
    with pytest.raises(ValueError):  # Assuming DataAnalysisTool raises ValueError for invalid operation
        await tool.execute(input_data)

@pytest.mark.asyncio
async def test_code_interpreter_tool():
    """Test CodeInterpreterTool execution with safe code."""
    tool = CodeInterpreterTool()
    code = "print(1 + 1)"
    result = await tool.execute(code)
    assert result == "2"

@pytest.mark.asyncio
async def test_code_interpreter_tool_unsafe_code():
    """Test CodeInterpreterTool with unsafe code."""
    tool = CodeInterpreterTool()
    code = "import os; os.remove('test.txt')"
    with pytest.raises(Exception):  # Assuming CodeInterpreterTool restricts unsafe operations
        await tool.execute(code)

@pytest.mark.asyncio
async def test_human_input_tool(mocker):
    """Test HumanInputTool with mocked input."""
    mocker.patch('builtins.input', return_value="user input")
    tool = HumanInputTool()
    result = await tool.execute("Test prompt")
    assert result == "user input"

@pytest.mark.asyncio
async def test_file_io_tool(tmp_path):
    """Test FileIOTool with temp files."""
    tool = FileIOTool()
    test_file = tmp_path / "test.txt"
    test_file.write_text("content")
    read_result = await tool.execute({"operation": "read", "path": str(test_file)})
    assert read_result == "content"
    write_result = await tool.execute({"operation": "write", "path": str(tmp_path / "write.txt"), "content": "new content"})
    assert write_result == "File written successfully"

@pytest.mark.asyncio
async def test_file_io_tool_invalid_path():
    """Test FileIOTool with invalid file path."""
    tool = FileIOTool()
    with pytest.raises(FileNotFoundError):
        await tool.execute({"operation": "read", "path": "/nonexistent/path.txt"})

@pytest.mark.asyncio
async def test_api_client_tool(mocker):
    """Test APIClientTool with mocked requests."""
    mocker.patch('requests.get', return_value=MagicMock(status_code=200, json=lambda: {"key": "value"}))
    tool = APIClientTool()
    input_data = {"url": "http://test.com", "method": "GET"}
    result = await tool.execute(input_data)
    assert result == {"key": "value"}

@pytest.mark.asyncio
async def test_api_client_tool_failure(mocker):
    """Test APIClientTool with failed request."""
    mocker.patch('requests.get', side_effect=requests.RequestException("API failure"))
    tool = APIClientTool()
    input_data = {"url": "http://test.com", "method": "GET"}
    with pytest.raises(requests.RequestException):
        await tool.execute(input_data)

@pytest.mark.asyncio
async def test_vector_retriever_tool(mocker):
    """Test VectorDBRetrieverTool with mocked Pinecone."""
    mocker.patch('pinecone.Pinecone', return_value=MagicMock(Index=MagicMock(query=MagicMock(return_value={"matches": [{"text": "match"}]}))))
    tool = VectorDBRetrieverTool(api_key="test-key")
    input_data = {"query": [0.1] * 1536}
    result = await tool.execute(input_data)
    assert isinstance(result, list)
    assert len(result) > 0

@pytest.mark.asyncio
async def test_vector_retriever_tool_invalid_query(mocker):
    """Test VectorDBRetrieverTool with invalid query vector."""
    mocker.patch('pinecone.Pinecone', return_value=MagicMock(Index=MagicMock(query=MagicMock(side_effect=ValueError("Invalid vector")))))
    tool = VectorDBRetrieverTool(api_key="test-key")
    input_data = {"query": [0.1] * 10}  # Invalid vector length
    with pytest.raises(ValueError):
        await tool.execute(input_data)

@pytest.mark.asyncio
async def test_contract_call_tool(mocker):
    """Test ContractCallTool with mocked web3."""
    mocker.patch('web3.Web3', return_value=MagicMock(eth=MagicMock(contract=MagicMock(functions=MagicMock(test_func=MagicMock(call=MagicMock(return_value="result")))))))
    tool = ContractCallTool(rpc_url="test", address="0x0", abi=[])
    input_data = {"function": "test_func", "args": []}
    result = await tool.execute(input_data)
    assert result == "result"

@pytest.mark.asyncio
async def test_contract_call_tool_invalid_function(mocker):
    """Test ContractCallTool with invalid function name."""
    mocker.patch('web3.Web3', return_value=MagicMock(eth=MagicMock(contract=MagicMock(functions=MagicMock()))))
    tool = ContractCallTool(rpc_url="test", address="0x0", abi=[])
    input_data = {"function": "nonexistent_func", "args": []}
    with pytest.raises(AttributeError):  # Assuming AttributeError for missing function
        await tool.execute(input_data)

@pytest.mark.asyncio
async def test_event_listener_tool(mocker):
    """Test EventListenerTool with mocked web3."""
    mocker.patch('web3.Web3', return_value=MagicMock(eth=MagicMock(get_logs=MagicMock(return_value=[{"event": "test"}]))))
    tool = EventListenerTool(rpc_url="test", address="0x0", abi=[])
    input_data = {"event_name": "TestEvent", "from_block": 1}
    result = await tool.execute(input_data)
    assert isinstance(result, list)
    assert len(result) > 0

@pytest.mark.asyncio
async def test_event_listener_tool_no_events(mocker):
    """Test EventListenerTool with no events."""
    mocker.patch('web3.Web3', return_value=MagicMock(eth=MagicMock(get_logs=MagicMock(return_value=[]))))
    tool = EventListenerTool(rpc_url="test", address="0x0", abi=[])
    input_data = {"event_name": "TestEvent", "from_block": 1}
    result = await tool.execute(input_data)
    assert isinstance(result, list)
    assert len(result) == 0

@pytest.mark.asyncio
async def test_oracle_fetch_tool(mocker):
    """Test OracleFetchTool with mocked contract call."""
    mocker.patch('web3.Web3', return_value=MagicMock(eth=MagicMock(contract=MagicMock(functions=MagicMock(get_price=MagicMock(call=MagicMock(return_value=1000)))))))
    tool = OracleFetchTool(rpc_url="test")
    input_data = {"oracle_type": "get_price", "params": []}
    result = await tool.execute(input_data)
    assert result == 1000

@pytest.mark.asyncio
async def test_oracle_fetch_tool_invalid_oracle(mocker):
    """Test OracleFetchTool with invalid oracle type."""
    mocker.patch('web3.Web3', return_value=MagicMock(eth=MagicMock(contract=MagicMock(functions=MagicMock()))))
    tool = OracleFetchTool(rpc_url="test")
    input_data = {"oracle_type": "invalid_oracle", "params": []}
    with pytest.raises(AttributeError):  # Assuming AttributeError for missing oracle function
        await tool.execute(input_data)

# Run tests if executed directly
if __name__ == "__main__":
    pytest.main()