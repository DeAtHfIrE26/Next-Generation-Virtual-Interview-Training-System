"""Integrity monitoring policy (E7), generalising the prototype's rules.

Prototype behaviour kept: a phone must be seen in 3 consecutive frames before a warning
(``PHONE_DETECT_THRESHOLD``); a second face is a violation; repeated voice mismatches can
end a session; one notice per episode (the prototype's ``previous_warning_message``).

Fixed: the prototype summed every warning type into one counter, so three unrelated events
(say, background noise twice and one lip-sync warning) could end a session meant for voice
mismatches. Here each event type has its own debounce, counter and limit, and the session
mode decides whether limits end the session:

- ``coaching`` (default, candidate practising alone): notices are shown and listed in the
  report as integrity notes; the session never ends because of them.
- ``proctored`` (opt-in strict practice, or B2B mode behind ``FEATURE_B2B_HIRING``): per-type
  limits end the session.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class EventType(StrEnum):
    PHONE = "phone"
    SECOND_PERSON = "second_person"
    NO_FACE = "no_face"
    FACE_MISMATCH = "face_mismatch"
    VOICE_MISMATCH = "voice_mismatch"
    SECOND_SPEAKER = "second_speaker"
    LIPSYNC_MISMATCH = "lipsync_mismatch"
    BACKGROUND_NOISE = "background_noise"


class Mode(StrEnum):
    COACHING = "coaching"
    PROCTORED = "proctored"


MESSAGES = {
    EventType.PHONE: "A phone is visible. Put it away to keep practice realistic.",
    EventType.SECOND_PERSON: "Another person is in view. Interviews are one-to-one.",
    EventType.NO_FACE: "We can't see your face. Centre yourself in the camera.",
    EventType.FACE_MISMATCH: "The face on camera doesn't match the enrolled user.",
    EventType.VOICE_MISMATCH: "The voice doesn't match the enrolled voice.",
    EventType.SECOND_SPEAKER: "A second voice was heard.",
    EventType.LIPSYNC_MISMATCH: "Your lip movement didn't match the audio for that answer.",
    EventType.BACKGROUND_NOISE: "Background noise is high. A quieter room helps the transcript.",
}


@dataclass(frozen=True)
class PolicyConfig:
    mode: Mode = Mode.COACHING
    # consecutive positive observations before an episode starts (frames or checks)
    debounce: dict[EventType, int] = field(
        default_factory=lambda: {
            EventType.PHONE: 3,
            EventType.SECOND_PERSON: 3,
            EventType.NO_FACE: 20,
            EventType.FACE_MISMATCH: 3,
            EventType.VOICE_MISMATCH: 1,
            EventType.SECOND_SPEAKER: 1,
            EventType.LIPSYNC_MISMATCH: 1,
            EventType.BACKGROUND_NOISE: 2,
        }
    )
    # episodes that end a proctored session
    limits: dict[EventType, int] = field(
        default_factory=lambda: {
            EventType.PHONE: 3,
            EventType.SECOND_PERSON: 2,
            EventType.FACE_MISMATCH: 2,
            EventType.VOICE_MISMATCH: 3,
            EventType.SECOND_SPEAKER: 3,
            EventType.LIPSYNC_MISMATCH: 3,
        }
    )


@dataclass(frozen=True)
class Notice:
    event: EventType
    t: float
    message: str
    episode: int
    end_session: bool


class IntegrityMonitor:
    def __init__(self, config: PolicyConfig | None = None):
        self.cfg = config or PolicyConfig()
        self._streak: dict[EventType, int] = dict.fromkeys(EventType, 0)
        self._active: dict[EventType, bool] = dict.fromkeys(EventType, False)
        self.episodes: dict[EventType, int] = dict.fromkeys(EventType, 0)
        self.log: list[Notice] = []

    def observe(self, event: EventType, present: bool, t: float) -> Notice | None:
        """Feed one observation. Returns a Notice when a new episode starts."""
        if not present:
            self._streak[event] = 0
            self._active[event] = False
            return None
        self._streak[event] += 1
        if self._active[event] or self._streak[event] < self.cfg.debounce.get(event, 1):
            return None
        self._active[event] = True
        self.episodes[event] += 1
        limit = self.cfg.limits.get(event)
        end = self.cfg.mode == Mode.PROCTORED and limit is not None and self.episodes[event] >= limit
        notice = Notice(event, t, MESSAGES[event], self.episodes[event], end)
        self.log.append(notice)
        return notice

    def single(self, event: EventType, t: float) -> Notice | None:
        """Report a per-utterance event (voice, lip-sync): a complete episode by itself."""
        notice = self.observe(event, True, t)
        self.observe(event, False, t)
        return notice

    def summary(self) -> dict[str, int]:
        return {e.value: n for e, n in self.episodes.items() if n}

    def to_state(self) -> dict:
        """JSON-serialisable state so a monitor survives across HTTP requests."""
        return {
            "streak": {e.value: n for e, n in self._streak.items() if n},
            "active": [e.value for e, a in self._active.items() if a],
            "episodes": self.summary(),
        }

    @classmethod
    def from_state(cls, state: dict | None, config: PolicyConfig | None = None) -> IntegrityMonitor:
        m = cls(config)
        state = state or {}
        for k, n in state.get("streak", {}).items():
            m._streak[EventType(k)] = n
        for k in state.get("active", []):
            m._active[EventType(k)] = True
        for k, n in state.get("episodes", {}).items():
            m.episodes[EventType(k)] = n
        return m
