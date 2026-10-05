"""Deterministic fallback question bank.

Used whenever the LLM is unavailable, times out, or returns something that fails validation.
Selection is deterministic for a given session seed, avoids repeats, and prefers the target
difficulty and the candidate's role family.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from functools import cache
from importlib import resources

from interview_core.nlp.roles import Family


@dataclass(frozen=True)
class BankQuestion:
    id: str
    category: str
    family: str
    difficulty: int
    competency: str
    question: str
    expected_points: tuple[str, ...]

    def as_question(self, rationale: str) -> dict:
        return {
            "question": self.question,
            "category": self.category,
            "difficulty": self.difficulty,
            "competency": self.competency,
            "rationale": rationale,
            "expected_points": list(self.expected_points),
        }


@cache
def load_bank() -> tuple[BankQuestion, ...]:
    raw = json.loads(
        resources.files("interview_core.nlp").joinpath("data/question_bank.json").read_text("utf-8")
    )
    return tuple(
        BankQuestion(
            q["id"],
            q["category"],
            q["family"],
            q["difficulty"],
            q["competency"],
            q["question"],
            tuple(q["expected_points"]),
        )
        for q in raw["questions"]
    )


def select(category: str, family: Family, difficulty: int, exclude: set[str], seed: str) -> BankQuestion:
    bank = [q for q in load_bank() if q.id not in exclude]
    if not bank:
        raise LookupError("question bank exhausted")

    def rank(q: BankQuestion) -> tuple[int, int, int]:
        cat_pen = 0 if q.category == category else 2
        fam_pen = 0 if q.family == family else (1 if q.family == "general" else 3)
        return (cat_pen + fam_pen, abs(q.difficulty - difficulty), 0)

    best = min(rank(q)[:2] for q in bank)
    pool = sorted((q for q in bank if rank(q)[:2] == best), key=lambda q: q.id)
    return random.Random(f"{seed}:{len(exclude)}").choice(pool)  # noqa: S311 - not security sensitive
