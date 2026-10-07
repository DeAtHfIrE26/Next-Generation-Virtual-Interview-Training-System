"""JSON Schemas for every LLM output. Only constructs supported by all configured providers'
structured-output modes are used (no numeric or length constraints; integers use ``enum``)."""

from __future__ import annotations

import json
from functools import cache
from importlib import resources


@cache
def load(name: str) -> dict:
    return json.loads(resources.files(__package__).joinpath(f"{name}.json").read_text(encoding="utf-8"))
