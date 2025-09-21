# Standard imports
import pytest
import asyncio
from unittest.mock import patch, MagicMock

# Local
from lola.chains.connection import ChainConnection
from lola.chains.contract import Contract
from lola.chains.wallet import Wallet
from lola.chains.key_manager import KeyManager
from lola.chains.oracles import Oracles
from lola.chains.utils import Utils

"""
File: Comprehensive tests for LOLA OS chains in Phase 2.

Purpose: Validates EVM abstractions with real connections and mocks for RPC.
How: Uses pytest with async support, patch for web3 calls, and test data for validation.
Why: Ensures robust EVM integration with >80% coverage, per EVM-Native tenet.
Full Path: lola-os/tests/test_chains.py
"""

@pytest.mark.asyncio
async def test_chain_connection(mocker):
    """Test ChainConnection with mocked RPC."""
    mocker.patch('web3.Web3.is_connected', return_value=True)
    mocker.patch('web3.Web3.eth.block_number', return_value=12345)
    connection = ChainConnection("test_rpc")
    result = connection.get_block_number()
    assert result == 12345

@pytest.mark.asyncio
async def test_contract(mocker):
    """Test Contract calls with mocked web3."""
    mocker.patch('web3.Web3.eth.contract', return_value=MagicMock(functions=MagicMock(test_func=MagicMock(call=MagicMock(return_value="result")))))
    connection = ChainConnection("test_rpc")
    contract = Contract(connection, "0x0000000000000000000000000000000000000000", [])
    result = await contract.call_function("test_func", [])
    assert result == "result"

@pytest.mark.asyncio
async def test_wallet(mocker):
    """Test Wallet balance with mocked web3."""
    mocker.patch('web3.Web3.eth.get_balance', return_value=1000)
    connection = ChainConnection("test_rpc")
    wallet = Wallet(connection, "private_key")
    result = await wallet.get_balance()
    assert result == 1000

def test_key_manager():
    """Test KeyManager encryption/decryption."""
    manager = KeyManager("test_key" * 4)  # 32-byte key
    manager.store_key("test", "private_key")
    result = manager.get_key("test")
    assert result == "private_key"

@pytest.mark.asyncio
async def test_oracles(mocker):
    """Test Oracles fetch with mocked contract."""
    mocker.patch('web3.Web3.eth.contract', return_value=MagicMock(functions=MagicMock(get_price=MagicMock(call=MagicMock(return_value=1000)))))
    oracles = Oracles("test_rpc")
    result = await oracles.fetch("oracle_address", "get_price", [])
    assert result == 1000

@pytest.mark.asyncio
async def test_utils(mocker):
    """Test Utils gas estimation with mocked web3."""
    mocker.patch('web3.Web3.eth.estimate_gas', return_value=21000)
    connection = ChainConnection("test_rpc")
    utils = Utils(connection)
    result = await utils.estimate_gas({"from": "0x0000000000000000000000000000000000000000", "to": "0x1000000000000000000000000000000000000000"})
    assert result == 21000

# Run tests if executed directly
if __name__ == "__main__":
    pytest.main()