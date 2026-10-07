"""The evaluator's JSON adapter over a chat model (E6 scoring)."""

from __future__ import annotations

import json

from interview_core.agent.json_provider import ChatJSONProvider
from interview_core.agent.llm import Usage
from interview_core.nlp import schemas


class Chat:
    name, model = "openai_compat", "m"

    def __init__(self, supports_schema: bool):
        self.supports_schema, self.kwargs = supports_schema, {}
        self.last_usage = Usage(10, 5)

    def stream(self, system, messages, **kw):
        self.kwargs = kw
        yield json.dumps({"ok": True})


def test_schema_capable_providers_get_grammar_constrained_output():
    """A CPU-hosted model spent a second full call repairing malformed evaluation JSON; providers that
    support constrained decoding now get the schema, so the reply always parses."""
    schema = schemas.load("evaluation")
    chat = Chat(supports_schema=True)
    ChatJSONProvider(chat).complete_json("sys", "user", schema, timeout_s=5)
    assert chat.kwargs["schema"]["required"] == schema["required"] and "$id" not in chat.kwargs["schema"]

    plain = Chat(supports_schema=False)
    ChatJSONProvider(plain).complete_json("sys", "user", schema, timeout_s=5)
    assert "schema" not in plain.kwargs


def test_a_rejected_schema_falls_back_to_unconstrained_output_once():
    from interview_core.agent import json_provider
    from interview_core.nlp.providers.base import TransientLLMError

    class Picky(Chat):
        model = "picky"

        def stream(self, system, messages, **kw):
            if "schema" in kw:
                raise TransientLLMError("openai_compat: HTTP 500")
            yield from super().stream(system, messages, **kw)

    chat = Picky(supports_schema=True)
    out = ChatJSONProvider(chat).complete_json("sys", "user", schemas.load("evaluation"), timeout_s=5)
    assert json.loads(out.text) == {"ok": True}
    assert ("openai_compat", "picky") in json_provider._SCHEMA_REJECTED
    ChatJSONProvider(chat).complete_json("sys", "user", schemas.load("evaluation"), timeout_s=5)
    assert "schema" not in chat.kwargs  # not asked again
