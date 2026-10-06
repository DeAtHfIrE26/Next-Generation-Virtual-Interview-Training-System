"""Interview parameters, blueprint and running state for the interviewer agent (E6).

Everything here is plain data (JSON round-trippable) so the session can be persisted after every
turn and resumed after a reconnect.
"""

from __future__ import annotations

import copy
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

InterviewType = Literal["technical", "behavioral", "system_design", "hr", "case", "mixed"]
Round = Literal["screening", "technical", "onsite", "final", "hr"]
SENIORITIES = ("intern", "junior", "mid", "senior", "lead", "principal")
INTERVIEW_TYPES = ("technical", "behavioral", "system_design", "hr", "case", "mixed")
ROUNDS = ("screening", "technical", "onsite", "final", "hr")
LANGUAGES = {"en": "English", "hi": "Hindi", "es": "Spanish", "fr": "French", "de": "German"}
START_DIFFICULTY = {"intern": 1, "junior": 2, "mid": 3, "senior": 4, "lead": 4, "principal": 5}


@dataclass
class InterviewParams:
    role: str
    seniority: str = "mid"
    company: str = ""
    company_style: str = ""
    job_description: str = ""
    resume_context: str = ""  # contact details already removed
    skills: list[str] = field(default_factory=list)
    interview_type: str = "mixed"
    round: str = "technical"
    difficulty: str = "auto"  # "auto" or "1".."5"
    language: str = "en"
    duration_minutes: int = 20
    persona: str = "maya"

    def __post_init__(self) -> None:
        self.role = self.role.strip()[:120]
        if self.seniority not in SENIORITIES:
            raise ValueError(f"seniority must be one of {SENIORITIES}")
        if self.interview_type not in INTERVIEW_TYPES:
            raise ValueError(f"interview_type must be one of {INTERVIEW_TYPES}")
        if self.round not in ROUNDS:
            raise ValueError(f"round must be one of {ROUNDS}")
        if self.difficulty != "auto" and self.difficulty not in {"1", "2", "3", "4", "5"}:
            raise ValueError("difficulty must be auto or 1-5")
        if self.language not in LANGUAGES:
            raise ValueError(f"language must be one of {sorted(LANGUAGES)}")
        self.duration_minutes = max(5, min(60, int(self.duration_minutes)))
        self.skills = [s.strip()[:60] for s in self.skills if s.strip()][:15]
        self.company, self.company_style = self.company.strip()[:120], self.company_style.strip()[:400]
        self.job_description, self.resume_context = self.job_description[:6000], self.resume_context[:6000]

    @property
    def start_difficulty(self) -> int:
        return int(self.difficulty) if self.difficulty != "auto" else START_DIFFICULTY[self.seniority]


@dataclass
class Competency:
    id: str
    name: str
    why: str
    weight: float
    minutes: float
    signals: list[str] = field(default_factory=list)


@dataclass
class Blueprint:
    summary: str
    competencies: list[Competency]
    opening: str
    style_notes: str
    start_difficulty: int
    emergency: bool = False  # True when built without an LLM (flagged in logs and UI)
    provider: str = ""

    def by_id(self, cid: str) -> Competency | None:
        return next((c for c in self.competencies if c.id == cid), None)


@dataclass
class Turn:
    index: int
    action: str  # open | follow_up | challenge | new_topic | revisit | wrap_up | close
    competency: str
    difficulty: int
    say: str  # what the interviewer said (the question)
    anchor_quote: str = ""  # words from the previous answer this question builds on
    reason: str = ""
    provider: str = ""
    emergency: bool = False
    asked_at: float = 0.0
    answer: str | None = None
    answer_words: list[dict[str, Any]] = field(default_factory=list)
    answer_seconds: float = 0.0
    answered_at: float | None = None
    score: int | None = None  # the agent's quick 1-5 read of this answer (set on the next turn)
    assessment: dict[str, Any] | None = None
    evaluation: dict[str, Any] | None = None  # full evidence-verified rubric (background job)
    corrections: list[str] = field(default_factory=list)  # code-enforced fixes to the agent's plan


@dataclass
class AgentState:
    session_id: str
    params: InterviewParams
    blueprint: Blueprint | None = None
    turns: list[Turn] = field(default_factory=list)
    difficulty: int = 3
    started_at: float = 0.0
    finished: bool = False
    events: list[dict[str, Any]] = field(default_factory=list)  # fallbacks, corrections (audit trail)

    @classmethod
    def new(cls, session_id: str, params: InterviewParams) -> AgentState:
        return cls(session_id, params, difficulty=params.start_difficulty)

    # ---- time
    def elapsed_s(self, now: float | None = None) -> float:
        return 0.0 if not self.started_at else (now or time.time()) - self.started_at

    def remaining_s(self, now: float | None = None) -> float:
        return max(0.0, self.params.duration_minutes * 60 - self.elapsed_s(now))

    # ---- coverage bookkeeping (deterministic, from recorded turns)
    def coverage(self) -> dict[str, dict[str, Any]]:
        cov: dict[str, dict[str, Any]] = {}
        if self.blueprint:
            for c in self.blueprint.competencies:
                cov[c.id] = {
                    "name": c.name,
                    "turns": 0,
                    "seconds": 0.0,
                    "scores": [],
                    "target_minutes": c.minutes,
                }
        for t in self.turns:
            if t.competency in cov:
                cov[t.competency]["turns"] += 1
                cov[t.competency]["seconds"] += t.answer_seconds
                if t.score is not None:
                    cov[t.competency]["scores"].append(t.score)
        return cov

    @property
    def awaiting_answer(self) -> Turn | None:
        return (
            self.turns[-1]
            if self.turns and self.turns[-1].answer is None and self.turns[-1].action != "close"
            else None
        )

    def log(self, kind: str, **data: Any) -> None:
        self.events.append({"t": round(time.time(), 3), "kind": kind, **data})

    # ---- persistence
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> AgentState:
        d = copy.deepcopy(d)  # never mutate the caller's dict (it may be parsed again)
        params = InterviewParams(**d.pop("params"))
        bp = d.pop("blueprint", None)
        blueprint = None
        if bp:
            comps = [Competency(**c) for c in bp.pop("competencies")]
            blueprint = Blueprint(competencies=comps, **bp)
        turns = [Turn(**t) for t in d.pop("turns", [])]
        return cls(params=params, blueprint=blueprint, turns=turns, **d)
