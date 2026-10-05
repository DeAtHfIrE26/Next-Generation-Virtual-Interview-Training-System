"""Claude via the official Anthropic Python SDK (optional extra: ``interview-core[anthropic]``).

- JSON-schema-constrained output with ``output_config.format`` (schema uses only supported
  constructs: no numeric/length constraints, ``additionalProperties: false`` everywhere).
- Server-side refusal fallback (``fallbacks: "default"``) so a classifier decline is retried
  on Anthropic's recommended model instead of failing the turn.
- Effort is set explicitly (this model's default is ``medium``).
"""

from __future__ import annotations

from interview_core.nlp.providers.base import LLMResponse, PermanentLLMError, RefusalError, TransientLLMError


class AnthropicProvider:
    name = "anthropic"
    DEFAULT_MODEL = "claude-opus-5-5"
    FALLBACK_BETA = "server-side-fallback-2026-07-01"

    def __init__(
        self, model: str = DEFAULT_MODEL, *, effort: str = "medium", max_tokens: int = 16000, client=None
    ):
        import anthropic  # optional dependency

        self._anthropic = anthropic
        # max_retries=0: retries are owned by StructuredLLM so they are counted and metered.
        self._client = client or anthropic.Anthropic(max_retries=0)
        self.model, self.effort, self.max_tokens = model, effort, max_tokens

    def complete_json(self, system: str, user: str, schema: dict, *, timeout_s: float) -> LLMResponse:
        a = self._anthropic
        try:
            resp = self._client.with_options(timeout=timeout_s).beta.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=system,
                messages=[{"role": "user", "content": user}],
                output_config={"effort": self.effort, "format": {"type": "json_schema", "schema": schema}},
                betas=[self.FALLBACK_BETA],
                fallbacks="default",
            )
        except (a.RateLimitError, a.APITimeoutError, a.APIConnectionError, a.InternalServerError) as e:
            raise TransientLLMError(f"anthropic: {type(e).__name__}") from e
        except a.APIStatusError as e:
            if e.status_code >= 500 or e.status_code in (408, 409, 429):
                raise TransientLLMError(f"anthropic: HTTP {e.status_code}") from e
            raise PermanentLLMError(f"anthropic: HTTP {e.status_code}") from e
        if resp.stop_reason == "refusal":
            raise RefusalError("anthropic: request declined")
        if resp.stop_reason == "max_tokens":
            raise TransientLLMError("anthropic: output truncated at max_tokens")
        text = next((b.text for b in resp.content if b.type == "text"), "")
        usage = getattr(resp, "usage", None)
        return LLMResponse(
            text,
            getattr(resp, "model", self.model),
            getattr(usage, "input_tokens", 0) or 0,
            getattr(usage, "output_tokens", 0) or 0,
        )
