# Standard imports
import typing as tp
from cryptography.fernet import Fernet

"""
File: Defines the KeyManager class for LOLA OS TMVP 1 Phase 2.

Purpose: Manages secure storage of private keys.
How: Uses cryptography for encryption/decryption.
Why: Ensures key security, per Radical Reliability tenet.
Full Path: lola-os/python/lola/chains/key_manager.py
Future Optimization: Migrate to Rust for hardware key storage (post-TMVP 1).
"""

class KeyManager:
    """KeyManager: Securely manages private keys."""

    def __init__(self, master_key: str):
        """
        Initialize with master key for encryption.

        Args:
            master_key: Master encryption key.
        """
        self.cipher = Fernet(master_key.encode())
        self.keys = {}

    def store_key(self, name: str, key: str) -> None:
        """
        Stores an encrypted key.

        Args:
            name: Key name.
            key: Private key to store.
        """
        encrypted = self.cipher.encrypt(key.encode())
        self.keys[name] = encrypted

    def get_key(self, name: str) -> str:
        """
        Retrieves and decrypts a key.

        Args:
            name: Key name.

        Returns:
            Decrypted key.

        Does Not: Handle key generation—use eth_account.
        """
        encrypted = self.keys.get(name)
        if not encrypted:
            raise ValueError(f"Key {name} not found.")
        return self.cipher.decrypt(encrypted).decode()

__all__ = ["KeyManager"]