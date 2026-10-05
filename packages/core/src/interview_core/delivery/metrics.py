"""Speaking pace, pauses and fillers, each tied to timestamps so feedback is verifiable."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from itertools import pairwise

from interview_core.speech.asr import Word

FILLER_WORDS = {"um", "uh", "uhm", "umm", "er", "erm", "ah", "hmm", "mm"}
PAUSE_S = 1.0


@dataclass
class DeliveryMetrics:
    words: int
    speaking_s: float
    words_per_minute: float | None
    pauses: list[dict] = field(default_factory=list)  # {"start": s, "duration": s}
    longest_pause_s: float = 0.0
    fillers: list[dict] = field(default_factory=list)  # {"word": w, "t": s}
    filler_per_100_words: float = 0.0
    measured: bool = True

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def compute(words: list[Word]) -> DeliveryMetrics:
    if len(words) < 2:
        return DeliveryMetrics(len(words), 0.0, None, measured=False)
    norm = [re.sub(r"[^a-z']", "", w.word.lower()) for w in words]
    span = words[-1].end - words[0].start
    pauses = []
    for a, b in pairwise(words):
        gap = b.start - a.end
        if gap >= PAUSE_S:
            pauses.append({"start": round(a.end, 2), "duration": round(gap, 2)})
    fillers = [
        {"word": n, "t": round(w.start, 2)} for n, w in zip(norm, words, strict=True) if n in FILLER_WORDS
    ]
    content = sum(1 for n in norm if n and n not in FILLER_WORDS)
    paused = sum(p["duration"] for p in pauses)
    speaking = max(span - paused, 1e-6)
    return DeliveryMetrics(
        words=content,
        speaking_s=round(span, 2),
        words_per_minute=round(content / speaking * 60, 1),
        pauses=pauses,
        longest_pause_s=max((p["duration"] for p in pauses), default=0.0),
        fillers=fillers,
        filler_per_100_words=round(100 * len(fillers) / max(content, 1), 2),
    )
