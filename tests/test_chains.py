# Standard imports
import pytest

# Local
from lola.chains import (
    ChainConnection, Contract, Wallet, KeyManager, Oracles, ChainUtils
)

"""
File: Tests for chains module in LOLA OS TMVP 1 Phase 2.

Purpose: Verifies chain component initialization and basic functionality.
How: Uses pytest to test chain classes.
Why: Ensures robust EVM interactions, per EVM-Native tenet.
Full Path: lola-os/tests/test_chains.py
"""
def test_chains_initialization():
    """Test initialization of chain components."""
    connection = ChainConnection("http://localhost")
    contract = Contract(connection, "0x0000000000000000000000000000000000000000", [])
    wallet = Wallet(connection, "0x0000000000000000000000000000000000000000")
    key_manager = KeyManager()
    oracles = Oracles(connection)
    utils = ChainUtils()

    assert isinstance(connection, ChainConnection)
    assert isinstance(contract, Contract)
    assert isinstance(wallet, Wallet)
    assert isinstance(key_manager, KeyManager)
    assert isinstance(oracles, Oracles)
    assert isinstance(utils, ChainUtils)

    assert isinstance(connection.is_connected(), bool)
    assert isinstance(contract.call_function("test"), dict)
    assert isinstance(wallet.get_balance(), dict)
    assert isinstance(key_manager.get_key("0x0000000000000000000000000000000000000000"), str)
    assert isinstance(oracles.fetch_data("test"), dict)
    assert isinstance(utils.estimate_gas(), dict)