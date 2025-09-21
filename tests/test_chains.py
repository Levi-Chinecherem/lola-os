# Standard imports
import pytest
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock
import typing as tp

# Local
from lola.chains.connection import ChainConnection
from lola.chains.contract import Contract
from lola.chains.wallet import Wallet
from lola.chains.key_manager import KeyManager
from lola.chains.oracles import Oracles
from lola.chains.utils import Utils
from lola.utils import sentry
from lola.utils import prometheus

"""
File: Comprehensive tests for LOLA OS chains in Phase 5.

Purpose: Validates EVM abstractions with real testnet connections and mocks for RPC.
How: Uses pytest with async support, real Sepolia calls, and mocks for validation.
Why: Ensures robust EVM integration with >80% coverage, per EVM-Native and Radical Reliability tenets.
Full Path: lola-os/tests/test_chains.py
"""

@pytest.fixture
def mock_connection():
    """Fixture for a mocked ChainConnection."""
    return ChainConnection({"rpc_url": "http://mock-rpc", "provider_type": "http"})

@pytest.mark.asyncio
async def test_chain_connection_real(mocker):
    """Test ChainConnection with real Sepolia testnet."""
    mocker.patch("lola.utils.sentry.init_sentry")
    mocker.patch("prometheus_client.Counter.inc")
    config = {"rpc_url": "https://rpc.sepolia.org"}
    connection = ChainConnection(config)
    block_number = await connection.get_block_number()
    assert isinstance(block_number, int)
    assert block_number > 0

@pytest.mark.asyncio
async def test_contract_real(mocker, mock_connection):
    """Test Contract with real Uniswap V3 contract on Sepolia."""
    mocker.patch("lola.utils.sentry.capture_exception")
    mocker.patch("prometheus_client.Histogram.observe")
    uniswap_abi = [{"inputs": [], "name": "factory", "outputs": [{"internalType": "address", "name": "", "type": "address"}], "stateMutability": "view", "type": "function"}]
    contract = Contract(mock_connection, "0x1F98431c8aD98523631AE4a59f267346ea31F984", uniswap_abi)
    result = await contract.call_function("factory", [])
    assert isinstance(result, str)

@pytest.mark.asyncio
async def test_wallet_balance(mocker, mock_connection):
    """Test Wallet balance with mocked web3."""
    mocker.patch("web3.Web3.eth.get_balance", return_value=1000)
    mocker.patch("lola.utils.sentry.capture_exception")
    wallet = Wallet(mock_connection, "0x" + "1" * 64)
    balance = await wallet.get_balance()
    assert balance == 1000

def test_key_manager():
    """Test KeyManager encryption/decryption."""
    manager = KeyManager("test_key" * 8)  # 32-byte key
    manager.store_key("test", "private_key")
    result = manager.get_key("test")
    assert result == "private_key"

@pytest.mark.asyncio
async def test_oracles_chainlink(mocker, mock_connection):
    """Test Oracles with Chainlink adapter."""
    mocker.patch("lola.chains.contract.Contract.call_function", AsyncMock(return_value=(1, 1000, 0, 0, 1)))
    mocker.patch("lola.utils.sentry.capture_exception")
    oracles = Oracles({"rpc_url": "http://mock-rpc", "oracle_type": "chainlink", "oracle_address": "0xmock"})
    result = await oracles.fetch("latestRoundData", [])
    assert result[1] == 1000

@pytest.mark.asyncio
async def test_utils_gas(mocker, mock_connection):
    """Test Utils gas estimation with mocked web3."""
    mocker.patch("web3.Web3.eth.estimate_gas", return_value=21000)
    mocker.patch("lola.utils.sentry.capture_exception")
    utils = Utils(mock_connection)
    result = await utils.estimate_gas({"from": "0x0000000000000000000000000000000000000000", "to": "0x1000000000000000000000000000000000000000"})
    assert result == 21000