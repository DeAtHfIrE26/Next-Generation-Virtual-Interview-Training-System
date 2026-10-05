"""Speaker verification with explicit enrolment state.

Fixes the prototype defect where, with no reference, the first interview answer silently
became the reference (``interview_core.legacy.voice.answer_voice_check``). Here an
un-enrolled session never matches anything: every utterance is reported ``not_enrolled``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol

import numpy as np

from interview_core.biometric import Template, VerifyResult, build_template, verify

MIN_ENROL_UTTERANCES = 3
MIN_ENROL_SPEECH_S = 8.0
MIN_PROBE_SPEECH_S = 1.5


class SpeakerEmbedder(Protocol):
    model_id: str

    def embed(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Embedding for one utterance (float32 mono samples in [-1, 1])."""
        ...


class VoiceStatus(StrEnum):
    NOT_ENROLLED = "not_enrolled"
    MATCH = "match"
    MISMATCH = "mismatch"
    UNCALIBRATED = "uncalibrated"  # scored, but no threshold configured: no decision
    TOO_SHORT = "too_short"


@dataclass
class UtteranceCheck:
    status: VoiceStatus
    score: float | None
    duration_s: float


class VoiceVerifier:
    def __init__(self, embedder: SpeakerEmbedder, threshold: float | None):
        self.embedder = embedder
        self.threshold = threshold

    def enrol(self, utterances: Sequence[tuple[np.ndarray, int]]) -> Template:
        total = sum(len(a) / sr for a, sr in utterances)
        if len(utterances) < MIN_ENROL_UTTERANCES or total < MIN_ENROL_SPEECH_S:
            raise ValueError(
                f"enrolment needs >= {MIN_ENROL_UTTERANCES} prompted phrases and "
                f">= {MIN_ENROL_SPEECH_S:.0f}s of speech (got {len(utterances)}, {total:.1f}s)"
            )
        embs = [self.embedder.embed(a, sr) for a, sr in utterances]
        return build_template("voice", self.embedder.model_id, embs, min_samples=MIN_ENROL_UTTERANCES)

    def verify(self, template: Template, audio: np.ndarray, sr: int) -> VerifyResult:
        return verify(template, self.embedder.embed(audio, sr), self.threshold, self.embedder.model_id)


@dataclass
class VoiceSession:
    """Real-time matching across an interview: one check per utterance."""

    verifier: VoiceVerifier
    template: Template | None
    checks: list[UtteranceCheck] = field(default_factory=list)

    def check(self, audio: np.ndarray, sr: int) -> UtteranceCheck:
        dur = len(audio) / sr
        if self.template is None:
            result = UtteranceCheck(VoiceStatus.NOT_ENROLLED, None, dur)
        elif dur < MIN_PROBE_SPEECH_S:
            result = UtteranceCheck(VoiceStatus.TOO_SHORT, None, dur)
        else:
            v = self.verifier.verify(self.template, audio, sr)
            status = (
                VoiceStatus.UNCALIBRATED
                if v.accepted is None
                else VoiceStatus.MATCH
                if v.accepted
                else VoiceStatus.MISMATCH
            )
            result = UtteranceCheck(status, v.score, dur)
        self.checks.append(result)
        return result

    @property
    def mismatch_count(self) -> int:
        return sum(c.status == VoiceStatus.MISMATCH for c in self.checks)
