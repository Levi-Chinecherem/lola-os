# Standard imports
import pytest
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from decimal import Decimal
import asyncio
from typing import Dict, Any
import time

# Local
from lola.libs.chainlink.price_feed import (
    LolaChainlinkPriceFeed, get_chainlink_price_feed,
    PriceFeedHealth, PriceFeedResult
)
from lola.utils.config import get_config
from lola.chains.connection import ChainConnection

"""
Test file for Chainlink price feed integration.
Purpose: Ensures price oracle queries work correctly with mocked web3 calls,
         proper health assessment, caching, and error handling.
Full Path: lola-os/tests/test_libs_chainlink_price_feed.py
"""

@pytest.fixture
def mock_config():
    """Mock configuration with Chainlink enabled."""
    with patch('lola.utils.config.get_config') as mock:
        mock.return_value = {
            "chainlink_enabled": True,
            "chainlink_default_chains": ["ethereum"],
            "rpc_endpoints": {
                1: "http://localhost:8545",
                137: "http://localhost:8546"
            },
            "chainlink_staleness_threshold": 1800,  # 30 minutes for testing
            "sentry_dsn": "test-dsn"
        }
        yield mock

@pytest.fixture
def price_feed(mock_config):
    """Fixture for LolaChainlinkPriceFeed."""
    return get_chainlink_price_feed()

@pytest.fixture
def mock_web3():
    """Mock Web3 connection."""
    mock_w3 = Mock()
    mock_contract = Mock()
    
    # Mock contract functions
    mock_contract.functions.latestRoundData.return_value.call.return_value = (
        123,  # roundId
        2000000000000000000000,  # answer (2000 * 10^18)
        1690000000,  # startedAt
        int(time.time()),  # updatedAt (now)
        123  # answeredInRound
    )
    
    mock_contract.functions.decimals.return_value.call.return_value = 18
    mock_contract.functions.description.return_value.call.return_value = "ETH / USD"
    
    mock_w3.eth.contract.return_value = mock_contract
    return mock_w3

@pytest.fixture
def mock_chain_connection(mock_web3):
    """Mock ChainConnection."""
    with patch('lola.chains.connection.ChainConnection') as mock_connection:
        mock_instance = Mock()
        mock_instance.get_contract.return_value = mock_instance
        mock_connection.return_value = mock_instance
        mock_instance._w3 = mock_web3
        yield mock_instance

async def test_price_query_success(price_feed, mock_chain_connection):
    """Test successful price query."""
    # Execute query
    result = await price_feed.get_price("ETH", "USD", "ethereum")
    
    # Verify result structure
    assert isinstance(result, PriceFeedResult)
    assert result.health == PriceFeedHealth.HEALTHY
    assert result.price is not None
    assert isinstance(result.price, Decimal)
    assert result.price > 0
    assert result.decimals == 18
    assert result.description == "ETH / USD"
    assert result.confidence >= 0.8
    
    # Verify contract interaction
    mock_chain_connection.get_contract.assert_called_once()

async def test_price_query_stale(price_feed, mock_chain_connection):
    """Test stale price detection."""
    # Mock old timestamp (2 hours ago)
    mock_round_data = MagicMock()
    mock_round_data.call.return_value = (
        123,
        2000000000000000000000,
        int(time.time() - 7200),  # 2 hours ago
        int(time.time() - 7200),
        123
    )
    mock_chain_connection.get_contract.return_value.functions.latestRoundData.return_value = mock_round_data
    
    result = await price_feed.get_price("ETH", "USD", "ethereum")
    
    assert result.health == PriceFeedHealth.STALE
    assert result.confidence < 1.0
    assert result.price is None  # Should not return stale price

async def test_price_query_low_confidence(price_feed, mock_chain_connection):
    """Test low confidence handling."""
    # Mock round mismatch
    mock_round_data = MagicMock()
    mock_round_data.call.return_value = (124, 2000000000000000000000,  # round_id != answered_in_round
                                        int(time.time()), int(time.time()), 123)
    mock_chain_connection.get_contract.return_value.functions.latestRoundData.return_value = mock_round_data
    
    result = await price_feed.get_price("ETH", "USD", "ethereum", min_confidence=0.9)
    
    assert result.health == PriceFeedHealth.DEGRADED
    assert result.confidence < 0.9

async def test_price_query_unavailable(price_feed, mock_chain_connection):
    """Test unavailable feed handling."""
    # Mock connection error
    mock_chain_connection.get_contract.side_effect = Exception("Contract not found")
    
    result = await price_feed.get_price("UNKNOWN", "USD", "ethereum")
    
    assert result.health == PriceFeedHealth.UNAVAILABLE
    assert result.price is None
    assert "UNKNOWN/USD feed unavailable" in result.description

async def test_price_caching(price_feed, mock_chain_connection):
    """Test price caching functionality."""
    # First query - should hit contract
    result1 = await price_feed.get_price("ETH", "USD", "ethereum")
    mock_chain_connection.get_contract.assert_called_once()
    
    # Reset call count
    mock_chain_connection.get_contract.call_count = 0
    
    # Second query - should hit cache
    result2 = await price_feed.get_price("ETH", "USD", "ethereum")
    assert mock_chain_connection.get_contract.call_count == 0  # No new calls
    
    # Verify same result
    assert result1.price == result2.price
    assert result1.timestamp == result2.timestamp

async def test_batch_price_query(price_feed, mock_chain_connection):
    """Test batch price querying."""
    pairs = ["ETH/USD", "BTC/USD"]
    
    results = await price_feed.batch_price_query(pairs, "ethereum")
    
    assert len(results) == 2
    assert "ETH/USD" in results
    assert "BTC/USD" in results
    assert all(isinstance(r, PriceFeedResult) for r in results.values())

def test_available_feeds_listing(price_feed):
    """Test available feeds listing."""
    feeds = price_feed.list_available_feeds("ethereum")
    
    assert "ETH/USD" in feeds
    assert feeds["ETH/USD"].startswith("0x")
    assert len(feeds) >= 2

def test_unknown_chain(price_feed, mock_chain_connection):
    """Test handling of unknown chains."""
    result = price_feed.get_price("ETH", "USD", "unknown-chain")
    
    assert result.health == PriceFeedHealth.UNAVAILABLE
    assert result.price is None

@patch('lola.libs.prometheus.exporter.get_lola_prometheus')
async def test_prometheus_integration(mock_prometheus, price_feed):
    """Test Prometheus metrics integration."""
    mock_exporter = Mock()
    mock_evm_call = Mock()
    mock_exporter.record_evm_call.return_value.__enter__.return_value = mock_evm_call
    mock_prometheus.return_value = mock_exporter
    
    await price_feed.get_price("ETH", "USD", "ethereum")
    
    # Verify EVM call metrics
    mock_exporter.record_evm_call.assert_called_once_with(
        chain="ethereum",
        operation="price_feed_ETH/USD"
    )

# Coverage marker
@pytest.mark.coverage
async def test_coverage_marker(price_feed):
    """Marker for coverage reporting."""
    pass

# Run with: pytest tests/test_libs_chainlink_price_feed.py -v --cov=lola/libs/chainlink --cov-report=html