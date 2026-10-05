"""Envelope encryption of biometric templates.

Each template gets a fresh 256-bit data key; the template is sealed with AES-256-GCM and the
data key is wrapped by a key-encryption key (KEK) held by a :class:`KeyProvider` (a cloud KMS
in production, an environment key in development). Associated data binds a ciphertext to its
owner and purpose, so a blob copied onto another user's row fails to decrypt.

Only templates (embedding vectors) are stored. Raw face images and voice recordings are never
persisted unless the user opts in (``RAW_MEDIA_RETENTION_DAYS``).
"""

from __future__ import annotations

import base64
import json
import os
import secrets
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from typing import Protocol

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from interview_core.biometric import Template


class KeyProvider(Protocol):
    key_id: str

    def wrap(self, data_key: bytes) -> bytes: ...

    def unwrap(self, wrapped: bytes) -> bytes: ...


class LocalKeyProvider:
    """KEK from ``TEMPLATE_KEK_BASE64``. Development and single-node use only."""

    def __init__(self, kek: bytes, key_id: str = "local-v1"):
        if len(kek) != 32:
            raise ValueError("KEK must be 32 bytes")
        self._aead = AESGCM(kek)
        self.key_id = key_id

    @classmethod
    def from_env(cls, var: str = "TEMPLATE_KEK_BASE64") -> LocalKeyProvider:
        raw = os.getenv(var, "")
        if not raw:
            raise RuntimeError(f"{var} is not set; refusing to store biometric templates unencrypted")
        return cls(base64.b64decode(raw))

    def wrap(self, data_key: bytes) -> bytes:
        nonce = secrets.token_bytes(12)
        return nonce + self._aead.encrypt(nonce, data_key, self.key_id.encode())

    def unwrap(self, wrapped: bytes) -> bytes:
        return self._aead.decrypt(wrapped[:12], wrapped[12:], self.key_id.encode())


@dataclass(frozen=True)
class EncryptedBlob:
    kek_id: str
    wrapped_key: str  # base64
    nonce: str  # base64
    ciphertext: str  # base64
    created_at: str

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, raw: str) -> EncryptedBlob:
        return cls(**json.loads(raw))


def _aad(
    owner_id: str, template: Template | None = None, kind: str | None = None, model_id: str | None = None
) -> bytes:
    kind = template.kind if template else kind
    model_id = template.model_id if template else model_id
    return f"{owner_id}|{kind}|{model_id}".encode()


def encrypt_template(template: Template, owner_id: str, keys: KeyProvider) -> EncryptedBlob:
    data_key = AESGCM.generate_key(bit_length=256)
    nonce = secrets.token_bytes(12)
    payload = json.dumps(
        {
            "kind": template.kind,
            "model_id": template.model_id,
            "n": template.n_samples,
            "created_at": template.created_at.isoformat(),
            "vector": base64.b64encode(template.to_bytes()).decode(),
        }
    ).encode()
    ct = AESGCM(data_key).encrypt(nonce, payload, _aad(owner_id, template))
    b64 = lambda b: base64.b64encode(b).decode()  # noqa: E731
    return EncryptedBlob(
        keys.key_id, b64(keys.wrap(data_key)), b64(nonce), b64(ct), template.created_at.isoformat()
    )


def decrypt_template(
    blob: EncryptedBlob, owner_id: str, kind: str, model_id: str, keys: KeyProvider
) -> Template:
    if blob.kek_id != keys.key_id:
        raise ValueError(f"blob sealed with KEK {blob.kek_id}, provider has {keys.key_id}")
    data_key = keys.unwrap(base64.b64decode(blob.wrapped_key))
    raw = AESGCM(data_key).decrypt(
        base64.b64decode(blob.nonce),
        base64.b64decode(blob.ciphertext),
        _aad(owner_id, kind=kind, model_id=model_id),
    )
    d = json.loads(raw)
    return Template.from_bytes(
        d["kind"],
        d["model_id"],
        base64.b64decode(d["vector"]),
        d["n"],
        datetime.fromisoformat(d["created_at"]),
    )


def is_expired(created_at: datetime, retention_days: int, now: datetime | None = None) -> bool:
    """Retention check; ``retention_days <= 0`` means delete at end of session (always expired)."""
    if retention_days <= 0:
        return True
    return (now or datetime.now(UTC)) >= created_at + timedelta(days=retention_days)
