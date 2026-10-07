"""Shared HTTP error mapping for raw-HTTP providers."""

from __future__ import annotations

import httpx

from interview_core.nlp.providers.base import PermanentLLMError, TransientLLMError


def post_json(url: str, headers: dict[str, str], body: dict, timeout_s: float, provider: str) -> dict:
    try:
        r = httpx.post(url, headers=headers, json=body, timeout=timeout_s)
    except (httpx.TimeoutException, httpx.TransportError) as e:
        raise TransientLLMError(f"{provider}: {type(e).__name__}") from e
    if r.status_code in (408, 409, 429) or r.status_code >= 500:
        raise TransientLLMError(f"{provider}: HTTP {r.status_code}")
    if r.status_code >= 400:
        raise PermanentLLMError(f"{provider}: HTTP {r.status_code}")
    return r.json()
