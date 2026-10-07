"""OpenAI Chat Completions with strict JSON-schema response format (raw HTTP)."""

from __future__ import annotations

from interview_core.nlp.providers.base import LLMResponse, RefusalError
from interview_core.nlp.providers.http_common import post_json


class OpenAIChatProvider:
    name = "openai"
    URL = "https://api.openai.com/v1/chat/completions"

    def __init__(self, model: str, api_key: str):
        self.model, self._key = model, api_key

    def complete_json(self, system: str, user: str, schema: dict, *, timeout_s: float) -> LLMResponse:
        body = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": schema.get("$id", "output").replace(".", "_"),
                    "schema": {k: v for k, v in schema.items() if k != "$id"},
                    "strict": True,
                },
            },
        }
        data = post_json(self.URL, {"Authorization": f"Bearer {self._key}"}, body, timeout_s, self.name)
        msg = data["choices"][0]["message"]
        if msg.get("refusal"):
            raise RefusalError("openai: request declined")
        usage = data.get("usage") or {}
        return LLMResponse(
            msg.get("content") or "",
            data.get("model", self.model),
            usage.get("prompt_tokens", 0),
            usage.get("completion_tokens", 0),
        )
