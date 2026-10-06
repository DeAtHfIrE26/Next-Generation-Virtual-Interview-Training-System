"""Adapter: use a streaming chat LLM wherever a ``complete_json`` provider is expected.

The answer evaluator (:mod:`interview_core.nlp.evaluator`) validates every JSON reply against its
schema and the candidate's own words, so one configured provider chain serves both the live
interviewer and the evidence-verified evaluation.
"""

from __future__ import annotations

import json

from interview_core.agent.llm import ChatLLM
from interview_core.nlp.providers.base import LLMResponse


class ChatJSONProvider:
    def __init__(self, chat: ChatLLM):
        self.chat = chat
        self.name, self.model = chat.name, chat.model

    def complete_json(self, system: str, user: str, schema: dict, *, timeout_s: float) -> LLMResponse:
        schema_text = json.dumps({k: v for k, v in schema.items() if k != "$id"}, separators=(",", ":"))
        sys = f"{system}\n\nReply with exactly one JSON object that conforms to this JSON Schema, and nothing else:\n{schema_text}"
        text = "".join(
            self.chat.stream(
                sys,
                [{"role": "user", "content": user}],
                max_tokens=2000,
                timeout_s=timeout_s,
                effort="medium",
            )
        )
        text = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        u = self.chat.last_usage
        return LLMResponse(text, self.chat.model, u.input_tokens, u.output_tokens)
