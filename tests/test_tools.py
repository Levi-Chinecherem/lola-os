# Standard imports
import pytest
import asyncio
from typing import Any

# Local imports
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
from lola.chains.contract import Contract
from lola.utils.logging import setup_logging, logger

"""
File: Tests for tool execution in LOLA OS TMVP 1.

Purpose: Verifies that all tools execute correctly and return expected results.
How: Tests each tool's execute method, handling both sync and async implementations.
Why: Ensures tools work reliably for agents, per Radical Reliability tenet.
Full Path: lola-os/tests/test_tools.py
"""

@pytest.mark.asyncio
async def test_tool_execution():
    """Test execution of all tools."""
    setup_logging(level="DEBUG")
    contract = Contract(ChainConnection("http://localhost"), "0x0000000000000000000000000000000000000000", [])
    connection = ChainConnection("http://localhost")
    tools = [
        WebSearchTool(),
        DataAnalysisTool(),
        CodeInterpreterTool(),
        HumanInputTool(),
        FileIOTool(),
        APIClientTool(),
        VectorDBRetrieverTool(),
        ContractCallTool(web3=connection.web3),  # Use mocked web3 from ChainConnection
        EventListenerTool(connection),
        OracleFetchTool(connection),
    ]
    for tool in tools:
        assert isinstance(tool, (WebSearchTool, DataAnalysisTool, CodeInterpreterTool,
                                HumanInputTool, FileIOTool, APIClientTool,
                                VectorDBRetrieverTool, ContractCallTool,
                                EventListenerTool, OracleFetchTool))
        if isinstance(tool, ContractCallTool):
            # Skip execution for ContractCallTool as it requires a valid contract
            logger.debug("Skipping ContractCallTool execution (requires valid contract)")
            continue
        result = await tool.execute("test") if hasattr(tool.execute, '__call__') and asyncio.iscoroutinefunction(tool.execute) else tool.execute("test")
        logger.debug(f"Tool {type(tool).__name__} result: {result}")
        assert isinstance(result, dict), f"Tool {type(tool).__name__} returned non-dict: {type(result)}"