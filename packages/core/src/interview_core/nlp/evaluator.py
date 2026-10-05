"""Answer evaluation with transformer-based models (E6), every score tied to evidence.

The LLM scores the answer on a published rubric (docs/RUBRIC.md) and must quote the
candidate's own words for each judgement. Quotes are verified against the transcript; an
evaluation citing words the candidate never said is rejected and repaired or replaced. When no
LLM is available the explainable heuristic evaluation is used and marked as such.

Scores are labelled "experimental" until the answer_scoring evaluation suite shows agreement
with human raters above the calibration gate (``SCORING_CALIBRATED``).
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any

from interview_core.nlp import heuristics, schemas
from interview_core.nlp.structured import CallRecord, StructuredLLM, wrap_untrusted

RUBRIC = """Score each dimension 1-5 using this rubric:
- relevance: 1 off-topic; 3 addresses the question partly; 5 fully answers what was asked.
- structure: 1 disorganised; 3 some order; 5 clear flow (for behavioural questions: situation, task, action, result).
- depth: 1 vague or generic; 3 some specifics; 5 concrete examples, numbers, trade-offs and reasoning.
- communication: 1 hard to follow; 3 understandable with effort; 5 concise and clear.
- technical_accuracy: null for non-technical questions; otherwise 1 incorrect; 3 partly correct; 5 correct and precise.
Judge only the words of the answer. Do not infer personality, emotions, nervousness, confidence
as a trait, accent, gender or any protected characteristic. Each evidence quote must be copied
exactly from the answer. Strengths and improvements must be specific and actionable."""

SYSTEM = (
    "You are an experienced interview coach giving fair, specific feedback on one answer in a "
    "practice interview. " + RUBRIC
)


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s']", " ", s.lower())).strip()


def quote_in_answer(quote: str, answer: str) -> bool:
    q = _norm(quote)
    return bool(q) and q in _norm(answer)


def evidence_check(answer: str):
    def check(data: dict[str, Any]) -> list[str]:
        errors = [
            f"evidence quote not found in the answer: {e['quote'][:60]!r}"
            for e in data["evidence"]
            if not quote_in_answer(e["quote"], answer)
        ]
        if len(heuristics.WORD.findall(answer)) >= 20 and not data["evidence"]:
            errors.append("provide at least one evidence quote from the answer")
        return errors

    return check


def calibrated() -> bool:
    return os.getenv("SCORING_CALIBRATED", "false").lower() == "true"


@dataclass
class Evaluation:
    data: dict[str, Any]
    method: str  # "llm" or "heuristic"
    overall: float  # 0..1
    calibrated: bool
    record: CallRecord

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.data,
            "method": self.method,
            "overall": self.overall,
            "label": "calibrated" if self.calibrated else "experimental",
        }


def overall_score(data: dict[str, Any]) -> float:
    vals = [v for v in data["scores"].values() if v is not None]
    return round((sum(vals) / len(vals) - 1) / 4, 3) if vals else 0.0


def evaluate(
    llm: StructuredLLM, question: dict[str, Any], answer: str, *, role: str, seniority: str
) -> Evaluation:
    schema = schemas.load("evaluation")
    user = (
        f"Role: {role} ({seniority})\nQuestion ({question['category']}, difficulty {question['difficulty']}): "
        f"{question['question']}\nWhat a strong answer covers: {'; '.join(question.get('expected_points', []))}\n\n"
        f"Candidate's answer (transcribed speech):\n{wrap_untrusted(answer)}"
    )
    if not answer.strip():
        offline = StructuredLLM(None)
        res = offline.generate(
            "evaluate",
            SYSTEM,
            user,
            schema,
            fallback=lambda: heuristics.heuristic_evaluation(question, answer),
        )
        llm.log.append(res.record)
    else:
        res = llm.generate(
            "evaluate",
            SYSTEM,
            user,
            schema,
            fallback=lambda: heuristics.heuristic_evaluation(question, answer),
            semantic=evidence_check(answer),
        )
    method = "heuristic" if res.record.used_fallback else "llm"
    return Evaluation(res.data, method, overall_score(res.data), calibrated() and method == "llm", res.record)
