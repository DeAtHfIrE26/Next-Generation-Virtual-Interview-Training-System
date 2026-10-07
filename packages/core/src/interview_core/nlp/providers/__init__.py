"""LLM provider adapters, selected by ``LLM_PROVIDER``. See :func:`from_env`."""

from __future__ import annotations

import os

from interview_core.nlp.providers.base import (
    LLMProvider,
    LLMResponse,
    PermanentLLMError,
    RefusalError,
    TransientLLMError,
)


def from_env() -> LLMProvider | None:
    kind = os.getenv("LLM_PROVIDER", "none").strip().lower()
    model = os.getenv("LLM_MODEL", "").strip()
    if kind in ("", "none"):
        return None
    if kind == "anthropic":
        from interview_core.nlp.providers.anthropic import AnthropicProvider

        return AnthropicProvider(
            model=model or AnthropicProvider.DEFAULT_MODEL, effort=os.getenv("LLM_EFFORT", "medium")
        )
    if kind == "openai":
        from interview_core.nlp.providers.openai_chat import OpenAIChatProvider

        return OpenAIChatProvider(
            model=_required(model), api_key=_required(os.getenv("OPENAI_API_KEY", ""), "OPENAI_API_KEY")
        )
    if kind == "mistral":
        from interview_core.nlp.providers.mistral import MistralProvider

        return MistralProvider(
            model=_required(model), api_key=_required(os.getenv("MISTRAL_API_KEY", ""), "MISTRAL_API_KEY")
        )
    raise RuntimeError(f"unknown LLM_PROVIDER={kind!r}")


def _required(value: str, name: str = "LLM_MODEL") -> str:
    if not value:
        raise RuntimeError(f"{name} must be set for this LLM_PROVIDER")
    return value


__all__ = ["LLMProvider", "LLMResponse", "PermanentLLMError", "RefusalError", "TransientLLMError", "from_env"]
