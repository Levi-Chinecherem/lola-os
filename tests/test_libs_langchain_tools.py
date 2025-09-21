# Standard imports
import pytest
from unittest.mock import Mock, patch
import typing as tp

# Local
from lola.libs.langchain.tools import LangChainToolsWrapper
from lola.tools.base import BaseTool
from lola.utils.config import get_config

"""
Test file for LangChain tools wrapper.
Purpose: Ensures LangChain tools can be properly wrapped and used in LOLA agents.
Full Path: lola-os/tests/test_libs_langchain_tools.py
"""

@pytest.fixture
def mock_config():
    """Mock config with LangChain enabled."""
    with patch('lola.utils.config.get_config') as mock:
        mock.return_value = {"use_langchain": True}
        yield mock

@pytest.fixture
def wrapper(mock_config):
    """Fixture for LangChainToolsWrapper."""
    return LangChainToolsWrapper()

def test_wrapper_initialization(wrapper):
    """Test wrapper initializes with available tools."""
    assert wrapper.enabled is True
    assert len(wrapper._available_tools) >= 1  # At least DuckDuckGo should be available

def test_get_wrapped_tool_success(wrapper):
    """Test successful tool wrapping for DuckDuckGo search."""
    tool = wrapper.get_wrapped_tool("duckduckgo_search")
    
    assert tool is not None
    assert isinstance(tool, BaseTool)
    assert tool.name == "duckduckgo_search"
    assert "search" in tool.description.lower()

def test_get_unknown_tool_returns_none(wrapper):
    """Test unknown tool returns None."""
    tool = wrapper.get_wrapped_tool("non_existent_tool")
    assert tool is None

def test_create_custom_tool_success(wrapper):
    """Test custom tool creation works."""
    def test_function(query: str) -> str:
        return f"Processed: {query}"
    
    tool = wrapper.create_custom_wrapped_tool(
        test_function, 
        "test_custom_tool", 
        "A test custom tool"
    )
    
    assert isinstance(tool, BaseTool)
    assert tool.name == "test_custom_tool"
    assert tool.execute("hello") == "Processed: hello"

def test_create_custom_tool_error_handling(wrapper):
    """Test custom tool creation handles errors."""
    def broken_function(query: str) -> str:
        raise ValueError("Broken function")
    
    with pytest.raises(ValueError, match="Failed to create custom LangChain tool"):
        wrapper.create_custom_wrapped_tool(
            broken_function,
            "broken_tool", 
            "Broken tool description"
        )

@patch('lola.libs.langchain.tools.LangChainToolsWrapper.enabled', False)
def test_disabled_wrapper():
    """Test wrapper behavior when disabled."""
    wrapper = LangChainToolsWrapper()
    tool = wrapper.get_wrapped_tool("duckduckgo_search")
    assert tool is None

def test_get_all_available_tools(wrapper):
    """Test getting list of available tools."""
    tools = wrapper.get_all_available_tools()
    assert isinstance(tools, list)
    assert "duckduckgo_search" in tools

# Integration test: Test actual search functionality (requires internet)
@pytest.mark.integration
@pytest.mark.skipif(not hasattr(pytest, 'mark'), reason="Integration test")
def test_search_tool_integration(wrapper):
    """Integration test for search tool (requires internet connection)."""
    tool = wrapper.get_wrapped_tool("duckduckgo_search")
    if tool:
        result = tool.execute("Python programming")
        assert isinstance(result, str)
        assert len(result) > 10  # Basic sanity check
    else:
        pytest.skip("Search tool not available")

# Coverage marker
@pytest.mark.coverage
def test_coverage_marker(wrapper):
    """Marker for coverage reporting."""
    pass

# Run with: pytest tests/test_libs_langchain_tools.py -v --cov=lola/libs/langchain/tools --cov-report=html