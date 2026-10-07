"""Map TTS viseme marks (or plain text, as a fallback) to the avatar's mouth shapes."""

from __future__ import annotations

import re

from interview_core.speech.tts import VisemeMark

# Avatar mouth shapes (drawn in apps/web/components/Avatar): rest, closed lips (P/B/M),
# lip-teeth (F/V), wide (E/I), open (A), round (O/U), tongue-teeth (TH/S/T/D/N/L), R/W.
MOUTH_SHAPES = ("rest", "mbp", "fv", "ee", "aa", "oo", "td", "rw")

POLLY = {
    "sil": "rest",
    "p": "mbp",
    "f": "fv",
    "T": "td",
    "t": "td",
    "s": "td",
    "S": "td",
    "k": "td",
    "i": "ee",
    "e": "ee",
    "E": "ee",
    "a": "aa",
    "@": "aa",
    "o": "oo",
    "O": "oo",
    "u": "oo",
    "r": "rw",
}

_LETTER = [
    ("th", "td"),
    ("sh", "td"),
    ("ch", "td"),
    ("oo", "oo"),
    ("ee", "ee"),
    ("ou", "oo"),
    ("a", "aa"),
    ("e", "ee"),
    ("i", "ee"),
    ("y", "ee"),
    ("o", "oo"),
    ("u", "oo"),
    ("w", "rw"),
    ("r", "rw"),
    ("m", "mbp"),
    ("b", "mbp"),
    ("p", "mbp"),
    ("f", "fv"),
    ("v", "fv"),
]


def from_polly(marks: list[VisemeMark]) -> list[tuple[int, str]]:
    return [(m.t_ms, POLLY.get(m.viseme, "td")) for m in marks]


def from_text(text: str, duration_ms: int) -> list[tuple[int, str]]:
    """Approximate timeline when the TTS gives no visemes (browser speech synthesis)."""
    shapes: list[str] = []
    for word in re.findall(r"[a-z']+", text.lower()):
        i = 0
        while i < len(word):
            for pat, shape in _LETTER:
                if word.startswith(pat, i):
                    shapes.append(shape)
                    i += len(pat)
                    break
            else:
                shapes.append("td")
                i += 1
        shapes.append("rest")
    if not shapes:
        return [(0, "rest")]
    step = duration_ms / len(shapes)
    return [(round(k * step), s) for k, s in enumerate(shapes)]
