"""Test doubles for LLM providers (tests only)."""

from __future__ import annotations

import json
from collections import deque

from interview_core.nlp.providers.base import LLMResponse


class ScriptedProvider:
    """Returns queued responses in order; items may be dicts, strings or exceptions."""

    name = "scripted"
    model = "scripted-1"

    def __init__(self, *responses):
        self.queue = deque(responses)
        self.calls: list[dict] = []

    def complete_json(self, system, user, schema, *, timeout_s):
        self.calls.append(
            {"system": system, "user": user, "schema": schema.get("$id"), "timeout_s": timeout_s}
        )
        item = self.queue.popleft() if self.queue else {}
        if isinstance(item, Exception):
            raise item
        text = item if isinstance(item, str) else json.dumps(item)
        return LLMResponse(text, self.model, 100, 50)
