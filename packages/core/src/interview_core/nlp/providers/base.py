"""Provider-neutral LLM interface used by the interviewer and evaluator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class TransientLLMError(RuntimeError):
    """Timeouts, rate limits, 5xx: worth retrying."""


class PermanentLLMError(RuntimeError):
    """Bad request, auth, unknown model: retrying will not help."""


class RefusalError(PermanentLLMError):
    """The model declined to answer."""


@dataclass(frozen=True)
class LLMResponse:
    text: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0


class LLMProvider(Protocol):
    name: str
    model: str

    def complete_json(self, system: str, user: str, schema: dict, *, timeout_s: float) -> LLMResponse:
        """Return text that should be a JSON document conforming to ``schema``.

        Implementations use the provider's native structured-output mode where available;
        callers validate regardless.
        """
        ...
