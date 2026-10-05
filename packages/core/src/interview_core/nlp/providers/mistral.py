"""Mistral La Plateforme chat completions in JSON mode (raw HTTP).

JSON mode guarantees syntactically valid JSON, not schema conformance, so the schema is also
given in the system prompt and the caller validates the result.
"""

from __future__ import annotations

import json

from interview_core.nlp.providers.base import LLMResponse
from interview_core.nlp.providers.http_common import post_json


class MistralProvider:
    name = "mistral"
    URL = "https://api.mistral.ai/v1/chat/completions"

    def __init__(self, model: str, api_key: str):
        self.model, self._key = model, api_key

    def complete_json(self, system: str, user: str, schema: dict, *, timeout_s: float) -> LLMResponse:
        sys_with_schema = f"{system}\n\nReturn only a JSON object that validates against this JSON Schema:\n{json.dumps(schema)}"
        body = {
            "model": self.model,
            "messages": [{"role": "system", "content": sys_with_schema}, {"role": "user", "content": user}],
            "response_format": {"type": "json_object"},
        }
        data = post_json(self.URL, {"Authorization": f"Bearer {self._key}"}, body, timeout_s, self.name)
        usage = data.get("usage") or {}
        return LLMResponse(
            data["choices"][0]["message"].get("content") or "",
            data.get("model", self.model),
            usage.get("prompt_tokens", 0),
            usage.get("completion_tokens", 0),
        )
