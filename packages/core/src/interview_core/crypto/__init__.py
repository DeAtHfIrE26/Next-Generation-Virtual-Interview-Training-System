"""Encryption at rest for biometric templates (envelope encryption, AES-256-GCM)."""

from interview_core.crypto.templates import (
    EncryptedBlob,
    KeyProvider,
    LocalKeyProvider,
    decrypt_template,
    encrypt_template,
    is_expired,
)

__all__ = [
    "EncryptedBlob",
    "KeyProvider",
    "LocalKeyProvider",
    "decrypt_template",
    "encrypt_template",
    "is_expired",
]
