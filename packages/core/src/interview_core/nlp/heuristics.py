"""Explainable answer features with character spans (E6 evaluation / E9 analysis).

Upgrade of the prototype's keyword counters (``interview_core.legacy.grading``): matches use
word boundaries (the prototype counted "um" inside "summary"), every signal is returned with
the character span it came from so the UI can highlight it, and the heuristic evaluation is
used only as the fallback when no LLM is available. It never judges technical correctness.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

FILLERS = (
    "um",
    "uh",
    "uhm",
    "umm",
    "er",
    "erm",
    "ah",
    "hmm",
    "you know",
    "i mean",
    "basically",
    "sort of",
    "kind of",
    "like i said",
)
HEDGES = ("maybe", "i guess", "i think", "probably", "perhaps", "not sure", "possibly", "i'm not certain")
STAR_CUES = {
    "situation": (
        "when i was",
        "at my previous",
        "in my last",
        "in my previous",
        "during my",
        "while working",
        "our team was",
        "the situation",
        "back when",
        "at the time",
        "we had a",
    ),
    "task": (
        "my role",
        "my responsibility",
        "i was responsible",
        "i needed to",
        "i had to",
        "the goal was",
        "the task was",
        "i was asked",
        "we needed to",
        "my job was",
    ),
    "action": (
        "i decided",
        "i implemented",
        "i built",
        "i created",
        "i designed",
        "i led",
        "i wrote",
        "i analysed",
        "i analyzed",
        "i organised",
        "i organized",
        "i proposed",
        "i set up",
        "i reached out",
        "i investigated",
        "i fixed",
        "i refactored",
        "so i",
        "i started",
        "i worked with",
    ),
    "result": (
        "as a result",
        "which led to",
        "this reduced",
        "this increased",
        "we reduced",
        "we increased",
        "resulted in",
        "the outcome",
        "in the end",
        "finally we",
        "we achieved",
        "we delivered",
        "saved",
        "improved by",
        "i learned",
    ),
}
NUMBER = re.compile(
    r"\b\d+(?:[.,]\d+)?\s*(?:%|percent|x|ms|seconds|minutes|hours|days|weeks|months|users|customers|k|m)?\b",
    re.I,
)
WORD = re.compile(r"[A-Za-z][A-Za-z'-]*")
STOP = frozenset(
    """a an the and or but if then so to of in on at for with by from as is are was were be been
being it this that these those i you he she we they me my our your their them his her its do did does doing
have has had what which who whom how why when where can could would should will shall may might must not no
yes about into over under again more most some such only own same than too very just also there here tell
describe walk time give example""".split()
)


@dataclass(frozen=True)
class Span:
    start: int
    end: int
    text: str
    label: str


@dataclass
class AnswerFeatures:
    words: int
    fillers: list[Span] = field(default_factory=list)
    hedges: list[Span] = field(default_factory=list)
    star: dict[str, list[Span]] = field(default_factory=dict)
    numbers: list[Span] = field(default_factory=list)

    @property
    def filler_rate(self) -> float:
        return len(self.fillers) / self.words if self.words else 0.0

    @property
    def star_coverage(self) -> int:
        return sum(1 for v in self.star.values() if v)


def _find(text: str, phrases: tuple[str, ...], label: str) -> list[Span]:
    out = []
    for p in phrases:
        for m in re.finditer(rf"(?<![A-Za-z']){re.escape(p)}(?![A-Za-z'])", text, re.I):
            out.append(Span(m.start(), m.end(), text[m.start() : m.end()], label))
    return sorted(out, key=lambda s: s.start)


def extract_features(answer: str) -> AnswerFeatures:
    return AnswerFeatures(
        words=len(WORD.findall(answer)),
        fillers=_find(answer, FILLERS, "filler"),
        hedges=_find(answer, HEDGES, "hedge"),
        star={k: _find(answer, cues, k) for k, cues in STAR_CUES.items()},
        numbers=[
            Span(m.start(), m.end(), m.group(0), "number")
            for m in NUMBER.finditer(answer)
            if m.group(0).strip()
        ],
    )


def content_words(text: str) -> set[str]:
    return {w.lower() for w in WORD.findall(text) if len(w) > 2 and w.lower() not in STOP}


def _band(value: float, cuts: tuple[float, ...]) -> int:
    """Map value to 1..5 using four ascending cut points."""
    return 1 + sum(value >= c for c in cuts)


def heuristic_evaluation(question: dict, answer: str) -> dict:
    """Schema-valid evaluation from observable features only (fallback path)."""
    f = extract_features(answer)
    target = content_words(question.get("question", "") + " " + " ".join(question.get("expected_points", [])))
    overlap = len(target & content_words(answer)) / max(1, min(len(target), 12))
    relevance = _band(overlap, (0.08, 0.2, 0.35, 0.5))
    structure = min(5, 1 + f.star_coverage)
    depth = min(5, _band(f.words, (25, 60, 110, 180)) + (1 if f.numbers else 0))
    communication = max(
        1, 5 - _band(f.filler_rate + 0.5 * len(f.hedges) / max(f.words, 1), (0.02, 0.04, 0.07, 0.1)) + 1
    )
    evidence = []
    for comp in ("situation", "action", "result"):
        if f.star[comp]:
            s = f.star[comp][0]
            evidence.append({"dimension": "structure", "quote": s.text, "comment": f"signals the {comp}"})
    if f.numbers:
        evidence.append({"dimension": "depth", "quote": f.numbers[0].text, "comment": "quantified detail"})
    if f.fillers:
        evidence.append(
            {
                "dimension": "communication",
                "quote": f.fillers[0].text,
                "comment": f"{len(f.fillers)} filler word(s) in {f.words} words",
            }
        )
    improvements = []
    if not f.star["result"]:
        improvements.append(
            "End with the outcome: what changed because of what you did, ideally with a number."
        )
    if not f.star["action"]:
        improvements.append("Say specifically what you did, using 'I' for your own actions.")
    if f.words < 40:
        improvements.append("Add one concrete example to support your point.")
    if f.fillers and f.filler_rate > 0.03:
        improvements.append("Pause briefly instead of using filler words.")
    strengths = []
    if f.star_coverage >= 3:
        strengths.append("Clear story structure (situation, actions and result).")
    if f.numbers:
        strengths.append("Uses concrete, quantified details.")
    need_follow = depth <= 2 or not f.star["result"]
    return {
        "scores": {
            "relevance": relevance,
            "structure": structure,
            "depth": depth,
            "communication": min(5, communication),
            "technical_accuracy": None,
        },
        "star": {k: bool(v) for k, v in f.star.items()},
        "evidence": evidence,
        "strengths": strengths,
        "improvements": improvements,
        "follow_up": {
            "needed": need_follow,
            "question": "Can you walk me through the specific outcome and what you personally did to get there?"
            if need_follow
            else "",
        },
        "summary": "Automatic feedback based on answer structure and delivery (content not assessed in this mode).",
    }
