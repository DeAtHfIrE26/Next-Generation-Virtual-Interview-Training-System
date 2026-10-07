"""Coding challenge bank with hidden test cases."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from functools import cache
from importlib import resources


@dataclass(frozen=True)
class TestCase:
    stdin: str
    expected: str
    hidden: bool


@dataclass(frozen=True)
class Challenge:
    id: str
    title: str
    prompt: str
    difficulty: int
    families: tuple[str, ...]
    languages: tuple[str, ...]
    tests: tuple[TestCase, ...]
    starter: dict[str, str]
    setup: str = ""

    def public(self) -> dict:
        """What the candidate may see: no hidden test cases."""
        return {
            "id": self.id,
            "title": self.title,
            "prompt": self.prompt,
            "difficulty": self.difficulty,
            "languages": list(self.languages),
            "starter": self.starter,
            "examples": [{"stdin": t.stdin, "expected": t.expected} for t in self.tests if not t.hidden],
            "hidden_tests": sum(t.hidden for t in self.tests),
        }


@cache
def load_challenges() -> tuple[Challenge, ...]:
    raw = json.loads(
        resources.files("interview_core.codeexec").joinpath("data/challenges.json").read_text("utf-8")
    )
    return tuple(
        Challenge(
            c["id"],
            c["title"],
            c["prompt"],
            c["difficulty"],
            tuple(c["family"]),
            tuple(c["languages"]),
            tuple(TestCase(t["stdin"], t["expected"], t["hidden"]) for t in c["tests"]),
            c.get("starter", {}),
            c.get("setup", ""),
        )
        for c in raw["challenges"]
    )


def pick_challenge(
    family: str, difficulty: int, seed: str, exclude: set[str] = frozenset()
) -> Challenge | None:
    pool = [c for c in load_challenges() if family in c.families and c.id not in exclude]
    if not pool:
        return None
    best = min(abs(c.difficulty - difficulty) for c in pool)
    pool = sorted((c for c in pool if abs(c.difficulty - difficulty) == best), key=lambda c: c.id)
    return random.Random(seed).choice(pool)  # noqa: S311
