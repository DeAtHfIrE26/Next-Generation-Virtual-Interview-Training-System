"""Encryption at rest for biometric templates (envelope encryption, AES-256-GCM)."""

from interview_core.crypto.templates import (
    EncryptedBlob,
    GcpKmsKeyProvider,
    KeyProvider,
    LocalKeyProvider,
    decrypt_template,
    encrypt_template,
    is_expired,
    key_provider_from_env,
)

__all__ = [
    "EncryptedBlob",
    "GcpKmsKeyProvider",
    "KeyProvider",
    "LocalKeyProvider",
    "decrypt_template",
    "encrypt_template",
    "is_expired",
    "key_provider_from_env",
]
