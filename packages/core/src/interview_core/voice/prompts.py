"""Prompted-phrase challenge: text-dependent protection against replayed recordings.

Each enrolment utterance (and optional re-verification) reads a freshly generated random
phrase. A recording made earlier cannot contain words that did not exist until now. The
spoken text is checked with ASR; the voice is checked by :class:`VoiceVerifier`.
"""

from __future__ import annotations

import secrets
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from interview_core.text import word_error_rate

# Short, unambiguous, accent-robust words (no homophones of each other).
WORDS = (
    "river",
    "orange",
    "window",
    "planet",
    "garden",
    "silver",
    "rocket",
    "pencil",
    "yellow",
    "marble",
    "forest",
    "candle",
    "button",
    "tiger",
    "violin",
    "harbor",
    "copper",
    "meadow",
    "lantern",
    "puzzle",
    "anchor",
    "basket",
    "falcon",
    "jungle",
    "kettle",
    "ladder",
    "museum",
    "number",
    "pillow",
    "rabbit",
    "saddle",
    "timber",
    "valley",
    "wagon",
    "zebra",
    "camera",
)
DIGITS = ("one", "two", "three", "four", "five", "six", "seven", "eight", "nine")


@dataclass(frozen=True)
class PhraseChallenge:
    nonce: str
    phrase: str
    expires_at: datetime


def issue_phrase(words: int = 4, digits: int = 2, ttl_s: float = 60.0) -> PhraseChallenge:
    rng = secrets.SystemRandom()
    parts = rng.sample(list(WORDS), words) + [rng.choice(DIGITS) for _ in range(digits)]
    rng.shuffle(parts)
    return PhraseChallenge(
        secrets.token_urlsafe(12), " ".join(parts), datetime.now(UTC) + timedelta(seconds=ttl_s)
    )


def check_phrase(
    challenge: PhraseChallenge, transcript: str, *, max_wer: float = 0.34, now: datetime | None = None
) -> tuple[bool, float]:
    """Return ``(ok, wer)``. Expired challenges always fail."""
    if (now or datetime.now(UTC)) > challenge.expires_at:
        return False, 1.0
    w = word_error_rate(challenge.phrase, transcript)
    return w <= max_wer, w
