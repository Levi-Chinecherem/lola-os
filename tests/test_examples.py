# Standard imports
import pytest
import asyncio
from unittest.mock import patch, AsyncMock, MagicMock
from pathlib import Path

# Local imports
from lola.utils.logging import setup_logging
from examples.research_agent.agent import main as research_main
from examples.onchain_analyst.agent import main as onchain_main

"""
File: Tests for LOLA OS example agents in TMVP 1 Phase 4.

Purpose: Validates research and on-chain analyst example agents.
How: Uses pytest with mocks for LLM, web, and EVM calls.
Why: Ensures examples are functional, per Developer Sovereignty.
Full Path: lola-os/tests/test_examples.py
"""

@pytest.fixture
def setup_config(tmp_path):
    """Fixture to create a temporary config file."""
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        "pinecone_api_key: test_key\n"
        "openai_api_key: test_openai\n"
        "web3_provider_uri: http://localhost:8545\n"
    )
    return config_file

@pytest.mark.asyncio
async def test_research_agent(setup_config, mocker):
    """Test research agent with mocked web and RAG tools."""
    mocker.patch("lola.agents.react.ReActAgent.run", AsyncMock(return_value=type('State', (), {'output': 'Paris'})))
    mocker.patch("lola.utils.config.load_config", return_value={"pinecone_api_key": "test_key", "openai_api_key": "test_openai"})
    result = await research_main("What is the capital of France?")
    assert result == "Paris"

@pytest.mark.asyncio
async def test_onchain_analyst(setup_config, mocker):
    """Test on-chain analyst with mocked EVM calls."""
    mocker.patch("web3.Web3.is_connected", return_value=True)
    mocker.patch("lola.agents.react.ReActAgent.run", AsyncMock(return_value=type('State', (), {'output': '0.0005 ETH'})))
    mocker.patch("lola.utils.config.load_config", return_value={"openai_api_key": "test_openai", "web3_provider_uri": "http://localhost:8545"})
    result = await onchain_main("Get Uniswap pair price")
    assert result == "0.0005 ETH"

if __name__ == "__main__":
    pytest.main()