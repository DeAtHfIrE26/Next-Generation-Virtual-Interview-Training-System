"""Text utilities shared by speech and NLP modules."""

from __future__ import annotations

import re

_PUNCT = re.compile(r"[^\w\s']")


def normalise_words(text: str) -> list[str]:
    return _PUNCT.sub(" ", text.lower()).split()


def word_error_rate(reference: str, hypothesis: str) -> float:
    ref, hyp = normalise_words(reference), normalise_words(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i] + [0] * len(hyp)
        for j, h in enumerate(hyp, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h))
        prev = cur
    return prev[-1] / len(ref)
