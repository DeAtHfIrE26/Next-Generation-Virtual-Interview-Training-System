"""Adaptive interviewer (E6): personalised questions conditioned on role, seniority, resume/JD and
previous answers, with per-answer difficulty adaptation and evidence-based follow-ups.

Prototype behaviour preserved: questions are generated from the resume and role by a
transformer model, with follow-ups based on the previous answer. Upgrades: a session plan
(category mix by role family), difficulty that moves with answer quality, duplicate detection,
schema validation of every question, and a deterministic question bank when the LLM is
unavailable or its output fails validation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from interview_core.nlp import question_bank, schemas
from interview_core.nlp.evaluator import Evaluation, evaluate
from interview_core.nlp.heuristics import content_words
from interview_core.nlp.roles import START_DIFFICULTY, Family, Seniority, is_technical, role_family
from interview_core.nlp.structured import StructuredLLM, wrap_untrusted

UP_AT, DOWN_BELOW = 0.75, 0.40
MAX_FOLLOW_UPS_PER_QUESTION = 1
DUPLICATE_JACCARD = 0.6

SYSTEM = (
    "You are a professional interviewer running a realistic practice interview. Ask exactly one "
    "question at a time, in a warm, neutral tone. Personalise it with the candidate's resume, the "
    "job description and earlier answers when relevant. Never ask about age, marital status, "
    "religion, caste, health, disability, pregnancy, nationality or other protected characteristics. "
    "Match the requested category and difficulty (1 = entry level, 5 = expert)."
)


def plan_for(family: Family, length: int = 8) -> list[str]:
    tech = is_technical(family)
    base = (
        ["behavioral", "technical", "technical", "behavioral", "technical", "situational", "role_specific"]
        if tech
        else [
            "behavioral",
            "role_specific",
            "behavioral",
            "situational",
            "technical",
            "behavioral",
            "situational",
        ]
    )
    body = (base * ((length // len(base)) + 1))[: max(1, length - 1)]
    return [*body, "wrap_up"]


def jaccard(a: str, b: str) -> float:
    wa, wb = content_words(a), content_words(b)
    return len(wa & wb) / len(wa | wb) if wa and wb else 0.0


@dataclass
class Turn:
    question: dict[str, Any]
    source: str  # "llm" | "bank" | "follow_up"
    bank_id: str | None = None
    parent: int | None = None  # index of the question this follows up
    answer: str | None = None
    evaluation: dict[str, Any] | None = None


@dataclass
class InterviewState:
    session_id: str
    role: str
    seniority: Seniority
    resume_context: str = ""
    job_description: str = ""
    length: int = 8
    family: Family = "general"
    difficulty: int = 3
    plan: list[str] = field(default_factory=list)
    plan_index: int = 0
    turns: list[Turn] = field(default_factory=list)

    @classmethod
    def start(
        cls,
        session_id: str,
        role: str,
        seniority: Seniority,
        resume_context: str = "",
        job_description: str = "",
        length: int = 8,
    ) -> InterviewState:
        fam = role_family(role)
        return cls(
            session_id,
            role,
            seniority,
            resume_context[:4000],
            job_description[:3000],
            length,
            fam,
            START_DIFFICULTY[seniority],
            plan_for(fam, length),
        )

    @property
    def finished(self) -> bool:
        return self.plan_index >= len(self.plan) and not self._pending_follow_up()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> InterviewState:
        turns = [Turn(**t) for t in d.pop("turns", [])]
        return cls(**d, turns=turns)

    def _pending_follow_up(self) -> Turn | None:
        if not self.turns:
            return None
        last = self.turns[-1]
        ev = last.evaluation or {}
        root = last.parent if last.parent is not None else len(self.turns) - 1
        n_follow = sum(1 for t in self.turns if t.parent == root)
        fu = ev.get("follow_up") or {}
        if fu.get("needed") and fu.get("question") and n_follow < MAX_FOLLOW_UPS_PER_QUESTION:
            return last
        return None


class Interviewer:
    def __init__(self, llm: StructuredLLM):
        self.llm = llm

    def next_question(self, st: InterviewState) -> Turn | None:
        if st.turns and st.turns[-1].answer is None:
            return st.turns[-1]  # still waiting for an answer: idempotent
        pending = st._pending_follow_up()
        if pending is not None:
            root = pending.parent if pending.parent is not None else len(st.turns) - 1
            q = dict(st.turns[root].question)
            q.update(
                question=pending.evaluation["follow_up"]["question"],
                rationale="Follow-up to get more detail on your previous answer.",
            )
            turn = Turn(q, "follow_up", parent=root)
            st.turns.append(turn)
            return turn
        if st.plan_index >= len(st.plan):
            return None
        category = st.plan[st.plan_index]
        st.plan_index += 1
        turn = self._generate(st, category)
        st.turns.append(turn)
        return turn

    def _generate(self, st: InterviewState, category: str) -> Turn:
        asked = [t.question["question"] for t in st.turns]
        bank_ids = {t.bank_id for t in st.turns if t.bank_id}
        last = next((t for t in reversed(st.turns) if t.answer), None)
        user = "\n".join(
            [
                f"Role: {st.role}\nSeniority: {st.seniority}\nCategory: {category}\nTarget difficulty: {st.difficulty}",
                "Resume (redacted):\n" + wrap_untrusted(st.resume_context or "(not provided)"),
                "Job description:\n" + wrap_untrusted(st.job_description or "(not provided)"),
                "Questions already asked (do not repeat or paraphrase):\n- " + "\n- ".join(asked)
                if asked
                else "",
                ("Previous answer summary: " + (last.evaluation or {}).get("summary", "")) if last else "",
                f"Return one {category} question at difficulty {st.difficulty}.",
            ]
        )

        def semantic(d: dict[str, Any]) -> list[str]:
            errs = []
            if d["category"] != category:
                errs.append(f"category must be {category}")
            if abs(d["difficulty"] - st.difficulty) > 1:
                errs.append(f"difficulty must be within 1 of {st.difficulty}")
            if any(jaccard(d["question"], a) >= DUPLICATE_JACCARD for a in asked):
                errs.append("question repeats an earlier question")
            if not d["question"].strip().endswith(("?", ".")) or len(d["question"]) > 400:
                errs.append("question must be one or two sentences")
            return errs

        picked: dict[str, Any] = {}

        def fallback() -> dict[str, Any]:
            bq = question_bank.select(category, st.family, st.difficulty, bank_ids, st.session_id)
            picked["id"] = bq.id
            return bq.as_question(f"Standard {bq.category.replace('_', ' ')} question for {st.role} roles.")

        res = self.llm.generate(
            "question", SYSTEM, user, schemas.load("question"), fallback=fallback, semantic=semantic
        )
        if res.record.used_fallback:
            return Turn(res.data, "bank", bank_id=picked.get("id"))
        return Turn(res.data, "llm")

    def submit_answer(self, st: InterviewState, answer: str) -> Evaluation:
        turn = st.turns[-1]
        if turn.answer is not None:
            raise ValueError("this question has already been answered")
        ev = evaluate(self.llm, turn.question, answer, role=st.role, seniority=st.seniority)
        turn.answer = answer
        turn.evaluation = ev.to_dict()
        if turn.source != "follow_up":
            st.difficulty = adapt_difficulty(st.difficulty, ev.overall)
        return ev


def adapt_difficulty(current: int, overall: float) -> int:
    if overall >= UP_AT:
        return min(5, current + 1)
    if overall < DOWN_BELOW:
        return max(1, current - 1)
    return current
