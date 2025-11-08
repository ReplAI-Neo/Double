#!/usr/bin/env python3
"""
Shared encryption utilities for message parsers.

Provides AES-256 symmetric encryption using Fernet.
"""

import base64
import secrets
from pathlib import Path
from typing import Optional


def generate_encryption_key() -> str:
    """
    Generate a secure random encryption key.

    Returns:
        Base64-encoded key string suitable for Fernet encryption
    """
    key = secrets.token_bytes(32)  # 256-bit key
    return base64.urlsafe_b64encode(key).decode("utf-8")


def encrypt_file(file_path: Path, encryption_key: str) -> None:
    """
    Encrypt a file using Fernet symmetric encryption (AES-256).

    Args:
        file_path: Path to the file to encrypt (will be overwritten)
        encryption_key: Base64-encoded encryption key

    Raises:
        ImportError: If cryptography library is not installed
        ValueError: If encryption key is invalid
    """
    try:
        from cryptography.fernet import Fernet
    except ImportError:
        raise ImportError(
            "cryptography library is required for encryption. "
            "Install it with: pip install cryptography"
        )

    # Ensure key is properly formatted
    key_bytes = encryption_key.encode("utf-8")
    cipher = Fernet(key_bytes)

    # Read the file
    with open(file_path, "rb") as f:
        file_data = f.read()

    # Encrypt the data
    encrypted_data = cipher.encrypt(file_data)

    # Write the encrypted data back (overwrite original)
    with open(file_path, "wb") as f:
        f.write(encrypted_data)


def decrypt_file(
    file_path: Path, encryption_key: str, output_path: Optional[Path] = None
) -> None:
    """
    Decrypt a file that was encrypted with Fernet symmetric encryption.

    Args:
        file_path: Path to the encrypted file
        encryption_key: Base64-encoded encryption key
        output_path: Optional path to save decrypted file
                    (defaults to same location with .decrypted extension)

    Raises:
        ImportError: If cryptography library is not installed
        ValueError: If decryption fails (wrong key or corrupted file)
    """
    try:
        from cryptography.fernet import Fernet
    except ImportError:
        raise ImportError(
            "cryptography library is required for decryption. "
            "Install it with: pip install cryptography"
        )

    # Ensure key is properly formatted
    key_bytes = encryption_key.encode("utf-8")
    cipher = Fernet(key_bytes)

    # Read the encrypted file
    with open(file_path, "rb") as f:
        encrypted_data = f.read()

    # Decrypt the data
    try:
        decrypted_data = cipher.decrypt(encrypted_data)
    except Exception as e:
        raise ValueError(f"Decryption failed. Invalid key or corrupted file: {e}")

    # Determine output path
    if output_path is None:
        if file_path.suffix == ".json":
            output_path = file_path.with_suffix(".decrypted.json")
        else:
            output_path = file_path.with_suffix(file_path.suffix + ".decrypted")

    # Write the decrypted data
    with open(output_path, "wb") as f:
        f.write(decrypted_data)

    return output_path
