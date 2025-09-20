# Standard imports
import pytest
import asyncio
from pathlib import Path

# Third-party imports
from web3 import Web3

# Local imports
from lola.utils.logging import setup_logging, logger
from examples.research_agent.agent import ResearchAgent
from examples.onchain_analyst.agent import OnchainAnalystAgent

"""
File: Tests for example agents in LOLA OS TMVP 1 Phase 4.

Purpose: Verifies functionality of research and onchain analyst agents.
How: Runs agents with mock queries and checks results.
Why: Ensures examples work for developers, per Developer Sovereignty.
Full Path: lola-os/tests/test_examples.py
"""

@pytest.mark.asyncio
async def test_research_agent():
    """Test the research agent example."""
    setup_logging(level="DEBUG")
    agent = ResearchAgent(tools=[])
    result = await agent.run("Test research query")
    logger.debug(f"Research agent result: {result.data}")
    assert result.data, "Research agent returned empty result"

@pytest.mark.asyncio
async def test_onchain_analyst_agent(monkeypatch):
    """Test the onchain analyst agent example."""
    setup_logging(level="DEBUG")
    # Mock Web3 to avoid real RPC calls
    class MockWeb3:
        def __init__(self, *args, **kwargs):
            self.eth = MockEth()
        def is_connected(self):
            return True
    class MockEth:
        def contract(self, *args, **kwargs):
            return None
    monkeypatch.setattr("web3.Web3", MockWeb3)
    agent = OnchainAnalystAgent(tools=[])
    result = await agent.run("Test onchain query")
    logger.debug(f"Onchain analyst result: {result.data}")
    assert result.data, "Onchain analyst returned empty result"