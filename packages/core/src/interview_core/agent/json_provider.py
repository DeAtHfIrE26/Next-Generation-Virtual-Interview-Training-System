"""Adapter: use a streaming chat LLM wherever a ``complete_json`` provider is expected.

The answer evaluator (:mod:`interview_core.nlp.evaluator`) validates every JSON reply against its
schema and the candidate's own words, so one configured provider chain serves both the live
interviewer and the evidence-verified evaluation.
"""

from __future__ import annotations

import json
import re

from interview_core.agent.llm import ChatLLM
from interview_core.nlp.providers.base import LLMResponse, PermanentLLMError, TransientLLMError

# (provider, model) pairs whose server could not compile a response schema into a grammar.
_SCHEMA_REJECTED: set[tuple[str, str]] = set()


class ChatJSONProvider:
    def __init__(self, chat: ChatLLM):
        self.chat = chat
        self.name, self.model = chat.name, chat.model

    def complete_json(self, system: str, user: str, schema: dict, *, timeout_s: float) -> LLMResponse:
        clean = {k: v for k, v in schema.items() if k != "$id"}
        schema_text = json.dumps(clean, separators=(",", ":"))
        sys = f"{system}\n\nReply with exactly one JSON object that conforms to this JSON Schema, and nothing else:\n{schema_text}"
        # Grammar-constrained decoding where the provider supports it: the reply always parses, so a
        # slow local model never spends a second full call repairing malformed JSON.
        use_schema = (
            getattr(self.chat, "supports_schema", False) and (self.name, self.model) not in _SCHEMA_REJECTED
        )
        constrained = {"schema": clean} if use_schema else {}

        def ask(extra: dict) -> str:
            return "".join(
                self.chat.stream(
                    sys,
                    [{"role": "user", "content": user}],
                    max_tokens=2000,
                    timeout_s=timeout_s,
                    effort="medium",
                    **extra,
                )
            )

        try:
            text = ask(constrained)
        except (PermanentLLMError, TransientLLMError) as e:
            # A server that cannot compile this schema into a grammar answers with an HTTP error, not
            # a timeout: ask once more unconstrained and stop sending the schema to this provider.
            if not constrained or not re.search(r"HTTP (400|422|500)\b", str(e)):
                raise
            _SCHEMA_REJECTED.add((self.name, self.model))
            text = ask({})
        text = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        u = self.chat.last_usage
        return LLMResponse(text, self.chat.model, u.input_tokens, u.output_tokens)
