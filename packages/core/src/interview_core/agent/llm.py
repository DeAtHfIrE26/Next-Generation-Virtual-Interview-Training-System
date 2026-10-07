"""Streaming chat LLMs for the interviewer agent.

Every provider exposes the same call: ``stream(system, messages, ...)`` yields text deltas as the
model produces them; ``last_usage`` holds token counts after the stream ends. Errors are mapped
to :class:`TransientLLMError` (retry or fail over), :class:`PermanentLLMError` (fail over) and
:class:`RefusalError`.

``schema`` (optional) asks providers that support it (``supports_schema = True``: Gemini and
OpenAI-compatible endpoints) for grammar-constrained JSON output. Anthropic models follow the tagged
reply format reliably, so the Anthropic provider ignores it and the agent accepts either format.

Providers:
- ``anthropic``: official SDK, streaming, prompt caching on the stable system prompt, server-side
  refusal fallback. Default model ``claude-opus-5-5`` at low effort for conversational turns.
- ``gemini``: Google Generative Language REST streaming (free tier available).
- ``openai_compat``: any OpenAI-compatible chat endpoint: Ollama (local, free), vLLM, LM Studio,
  Groq, OpenRouter, Together, OpenAI, Mistral.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Protocol

import httpx

from interview_core.nlp.providers.base import PermanentLLMError, RefusalError, TransientLLMError


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0


class ChatLLM(Protocol):
    name: str
    model: str
    last_usage: Usage

    def stream(
        self,
        system: str,
        messages: list[dict[str, str]],
        *,
        max_tokens: int = 1200,
        timeout_s: float = 30.0,
        effort: str = "low",
        schema: dict | None = None,
    ) -> Iterator[str]: ...


def _http_error(provider: str, status: int) -> Exception:
    if status in (408, 409, 429) or status >= 500:
        return TransientLLMError(f"{provider}: HTTP {status}")
    return PermanentLLMError(f"{provider}: HTTP {status}")


def _sse_lines(r: httpx.Response) -> Iterator[str]:
    for line in r.iter_lines():
        if line.startswith("data:"):
            data = line[5:].strip()
            if data and data != "[DONE]":
                yield data


# ----------------------------------------------------------------------------- Anthropic


class AnthropicChat:
    name = "anthropic"
    DEFAULT_MODEL = "claude-opus-5-5"
    FALLBACK_BETA = "server-side-fallback-2026-07-01"

    def __init__(self, model: str | None = None, client=None):
        import anthropic  # optional extra: interview-core[anthropic]

        self._a = anthropic
        # Retries are owned by the agent (counted, logged, then provider fail-over).
        self._client = client or anthropic.Anthropic(max_retries=0)
        self.model = model or self.DEFAULT_MODEL
        self.last_usage = Usage()

    def stream(self, system, messages, *, max_tokens=1200, timeout_s=30.0, effort="low", schema=None):
        a = self._a
        self.last_usage = Usage()
        try:
            with self._client.with_options(timeout=timeout_s).beta.messages.stream(
                model=self.model,
                max_tokens=max_tokens,
                # The system prompt is identical for every turn of every session: cache it.
                system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
                messages=messages,
                output_config={"effort": effort},
                betas=[self.FALLBACK_BETA],
                fallbacks="default",
            ) as s:
                yield from s.text_stream
                final = s.get_final_message()
        except (a.RateLimitError, a.APITimeoutError, a.APIConnectionError, a.InternalServerError) as e:
            raise TransientLLMError(f"anthropic: {type(e).__name__}") from e
        except a.APIStatusError as e:
            raise _http_error("anthropic", e.status_code) from e
        if final.stop_reason == "refusal":
            raise RefusalError("anthropic: request declined")
        if final.stop_reason == "max_tokens":
            raise TransientLLMError("anthropic: output truncated at max_tokens")
        u = final.usage
        self.last_usage = Usage(
            getattr(u, "input_tokens", 0) or 0,
            getattr(u, "output_tokens", 0) or 0,
            getattr(u, "cache_read_input_tokens", 0) or 0,
        )


# ----------------------------------------------------------------------------- Gemini


class GeminiChat:
    supports_schema = True
    name = "gemini"
    DEFAULT_MODEL = "gemini-2.5-flash"
    URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:streamGenerateContent"

    def __init__(self, api_key: str, model: str | None = None):
        self._key, self.model = api_key, model or self.DEFAULT_MODEL
        self.last_usage = Usage()

    def stream(self, system, messages, *, max_tokens=1200, timeout_s=30.0, effort="low", schema=None):
        self.last_usage = Usage()
        body = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [
                {"role": "model" if m["role"] == "assistant" else "user", "parts": [{"text": m["content"]}]}
                for m in messages
            ],
            "generationConfig": {"maxOutputTokens": max_tokens, "temperature": 0.7},
        }
        if effort == "low":
            body["generationConfig"]["thinkingConfig"] = {"thinkingBudget": 0}
        if schema is not None:
            body["generationConfig"]["responseMimeType"] = "application/json"
            body["generationConfig"]["responseJsonSchema"] = schema
        try:
            with httpx.stream(
                "POST",
                self.URL.format(model=self.model),
                params={"alt": "sse"},
                headers={"x-goog-api-key": self._key},
                json=body,
                timeout=timeout_s,
            ) as r:
                if r.status_code >= 400:
                    r.read()
                    raise _http_error("gemini", r.status_code)
                for data in _sse_lines(r):
                    d = json.loads(data)
                    for cand in d.get("candidates", []):
                        if cand.get("finishReason") in ("SAFETY", "PROHIBITED_CONTENT", "BLOCKLIST"):
                            raise RefusalError("gemini: response blocked")
                        for part in cand.get("content", {}).get("parts", []):
                            if part.get("text") and not part.get("thought"):
                                yield part["text"]
                    if um := d.get("usageMetadata"):
                        self.last_usage = Usage(
                            um.get("promptTokenCount", 0),
                            um.get("candidatesTokenCount", 0),
                            um.get("cachedContentTokenCount", 0),
                        )
        except (httpx.TimeoutException, httpx.TransportError) as e:
            raise TransientLLMError(f"gemini: {type(e).__name__}") from e


# ----------------------------------------------------------------------------- OpenAI-compatible


class OpenAICompatChat:
    """Chat Completions streaming against any compatible endpoint (Ollama, Groq, OpenRouter, ...)."""

    supports_schema = True

    def __init__(self, base_url: str, model: str, api_key: str = "", name: str = "openai_compat"):
        self.base_url, self.model, self._key, self.name = base_url.rstrip("/"), model, api_key, name
        self.last_usage = Usage()

    def stream(self, system, messages, *, max_tokens=1200, timeout_s=30.0, effort="low", schema=None):
        self.last_usage = Usage()
        body = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, *messages],
            "max_tokens": max_tokens,
            "temperature": 0.7,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if schema is not None:
            # Grammar-constrained decoding (Ollama >= 0.5, vLLM, OpenAI): the reply is guaranteed to
            # parse as JSON matching the schema, which removes format failures on small local models.
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "reply", "schema": schema, "strict": True},
            }
        headers = {"Authorization": f"Bearer {self._key}"} if self._key else {}
        try:
            with httpx.stream(
                "POST", f"{self.base_url}/chat/completions", headers=headers, json=body, timeout=timeout_s
            ) as r:
                if r.status_code >= 400:
                    r.read()
                    raise _http_error(self.name, r.status_code)
                for data in _sse_lines(r):
                    d = json.loads(data)
                    for ch in d.get("choices", []):
                        delta = ch.get("delta") or {}
                        if delta.get("refusal"):
                            raise RefusalError(f"{self.name}: request declined")
                        if delta.get("content"):
                            yield delta["content"]
                    if u := d.get("usage"):
                        self.last_usage = Usage(u.get("prompt_tokens", 0), u.get("completion_tokens", 0))
        except (httpx.TimeoutException, httpx.TransportError) as e:
            raise TransientLLMError(f"{self.name}: {type(e).__name__}") from e


# ----------------------------------------------------------------------------- selection


def build(kind: str, model: str = "") -> ChatLLM:
    kind = kind.strip().lower()
    if kind == "anthropic":
        return AnthropicChat(model or None)
    if kind == "gemini":
        key = os.getenv("GEMINI_API_KEY", "")
        if not key:
            raise PermanentLLMError("LLM provider gemini needs GEMINI_API_KEY")
        return GeminiChat(key, model or None)
    if kind in ("openai_compat", "ollama"):
        return OpenAICompatChat(
            os.getenv("OPENAI_COMPAT_BASE_URL", "http://localhost:11434/v1"),
            model or os.getenv("OPENAI_COMPAT_MODEL", "qwen2.5:7b-instruct"),
            os.getenv("OPENAI_COMPAT_API_KEY", ""),
            "ollama" if kind == "ollama" else "openai_compat",
        )
    if kind == "openai":
        key = os.getenv("OPENAI_API_KEY", "")
        if not key:
            raise PermanentLLMError("LLM provider openai needs OPENAI_API_KEY")
        return OpenAICompatChat("https://api.openai.com/v1", model or "gpt-4.1-mini", key, "openai")
    if kind == "mistral":
        key = os.getenv("MISTRAL_API_KEY", "")
        if not key:
            raise PermanentLLMError("LLM provider mistral needs MISTRAL_API_KEY")
        return OpenAICompatChat("https://api.mistral.ai/v1", model or "mistral-small-latest", key, "mistral")
    raise PermanentLLMError(f"unknown LLM provider {kind!r}")


def chain_from_env() -> list[ChatLLM]:
    """Primary provider then the optional fallback provider (``LLM_FALLBACK_PROVIDER``)."""
    chain: list[ChatLLM] = []
    primary = os.getenv("LLM_PROVIDER", "").strip()
    if primary and primary != "none":
        chain.append(build(primary, os.getenv("LLM_MODEL", "")))
    fallback = os.getenv("LLM_FALLBACK_PROVIDER", "").strip()
    if fallback and fallback != "none" and fallback != primary:
        chain.append(build(fallback, os.getenv("LLM_FALLBACK_MODEL", "")))
    return chain
