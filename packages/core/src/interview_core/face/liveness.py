"""Active liveness: randomised challenge-response from face-landmark series (E2 hardening).

The server issues a single-use challenge (random order of 3 of: blink, turn left, turn
right, open mouth) that expires quickly. The client records face landmarks while the user
performs it and submits the series. A printed photo cannot blink or turn; a pre-recorded
video cannot follow an order it did not know in advance. Passive presentation-attack
detection (screen and mask artefacts) needs a trained model and is a separate, optional
component (``PassivePAD``).

Landmark coordinates are MediaPipe face-mesh normalised (x, y) in the **unmirrored** camera
image. When the user turns their head to *their* left, the nose moves toward image right.
"""

from __future__ import annotations

import hashlib
import secrets
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any, Literal, Protocol

import numpy as np

from interview_core.lipsync.mouth import mouth_aperture

Step = Literal["blink", "turn_left", "turn_right", "open_mouth"]
ALL_STEPS: tuple[Step, ...] = ("blink", "turn_left", "turn_right", "open_mouth")

LEFT_EYE = (33, 160, 158, 133, 153, 144)
RIGHT_EYE = (362, 385, 387, 263, 373, 380)
NOSE_TIP = 1
EYE_OUTER = (33, 263)


@dataclass(frozen=True)
class LivenessConfig:
    steps: int = 3
    ttl_s: float = 30.0
    min_fps: float = 8.0
    min_face_ratio: float = 0.9
    blink_drop: float = 0.65  # EAR below this fraction of baseline counts as closed
    blink_recover: float = 0.85
    yaw_delta: float = 0.12  # nose offset / inter-ocular distance, relative to baseline
    mouth_delta: float = 0.20  # aperture increase over baseline


@dataclass(frozen=True)
class Challenge:
    nonce: str
    steps: tuple[Step, ...]
    issued_at: datetime
    expires_at: datetime

    def to_dict(self) -> dict[str, Any]:
        return {
            "nonce": self.nonce,
            "steps": list(self.steps),
            "issued_at": self.issued_at.isoformat(),
            "expires_at": self.expires_at.isoformat(),
        }


@dataclass
class LivenessResult:
    passed: bool
    score: float  # fraction of required steps completed in order
    detected: list[dict[str, Any]] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)


class PassivePAD(Protocol):
    model_id: str

    def score(self, face_rgb: np.ndarray) -> float:
        """Probability-like score that the crop is a live face (higher = live)."""
        ...


def issue_challenge(now: datetime | None = None, config: LivenessConfig | None = None) -> Challenge:
    cfg = config or LivenessConfig()
    rng = secrets.SystemRandom()
    steps = tuple(rng.sample(list(ALL_STEPS), cfg.steps))
    now = now or datetime.now(UTC)
    return Challenge(secrets.token_urlsafe(16), steps, now, now + timedelta(seconds=cfg.ttl_s))


def _pts(lm: Any) -> np.ndarray:
    return np.asarray(lm, dtype=np.float64)[:, :2]


def eye_aspect_ratio(lm: np.ndarray) -> float:
    def ear(idx: tuple[int, ...]) -> float:
        p = lm[list(idx)]
        return (np.linalg.norm(p[1] - p[5]) + np.linalg.norm(p[2] - p[4])) / (
            2 * np.linalg.norm(p[0] - p[3]) + 1e-9
        )

    return (ear(LEFT_EYE) + ear(RIGHT_EYE)) / 2


def yaw_proxy(lm: np.ndarray) -> float:
    mid = (lm[EYE_OUTER[0]] + lm[EYE_OUTER[1]]) / 2
    iod = np.linalg.norm(lm[EYE_OUTER[0]] - lm[EYE_OUTER[1]]) + 1e-9
    return float((lm[NOSE_TIP][0] - mid[0]) / iod)


def series_digest(series: dict[str, Any]) -> str:
    """Stable digest of a submitted series, for rejecting byte-for-byte resubmissions."""
    h = hashlib.sha256()
    for fr in series.get("frames", []):
        h.update(np.round(np.asarray(fr.get("landmarks") or [], dtype=np.float64), 4).tobytes())
        h.update(str(round(float(fr.get("t", 0)), 3)).encode())
    return h.hexdigest()


def _detect_steps(
    times: np.ndarray,
    ear: np.ndarray,
    yaw: np.ndarray,
    mouth: np.ndarray,
    wanted: tuple[Step, ...],
    cfg: LivenessConfig,
) -> list[dict[str, Any]]:
    base_n = max(3, int(np.searchsorted(times, times[0] + 0.5)))
    b_ear, b_yaw, b_mouth = np.median(ear[:base_n]), np.median(yaw[:base_n]), np.median(mouth[:base_n])
    detected: list[dict[str, Any]] = []
    i = base_n
    for step in wanted:
        hit = None
        while i < times.size and hit is None:
            if step == "blink" and ear[i] < cfg.blink_drop * b_ear:
                j = i
                while j < times.size and times[j] - times[i] <= 0.6:
                    if ear[j] > cfg.blink_recover * b_ear:
                        hit = (i, j)
                        break
                    j += 1
            elif step == "turn_left" and yaw[i] - b_yaw > cfg.yaw_delta:
                hit = (i, i)
            elif step == "turn_right" and b_yaw - yaw[i] > cfg.yaw_delta:
                hit = (i, i)
            elif step == "open_mouth" and mouth[i] - b_mouth > cfg.mouth_delta:
                hit = (i, i)
            i += 1
        if hit is None:
            break
        detected.append({"step": step, "t": float(times[hit[0]])})
        # Require a return toward neutral before the next step so one pose cannot satisfy two.
        k = hit[1]
        while k < times.size and (
            abs(yaw[k] - b_yaw) > cfg.yaw_delta / 2 or mouth[k] - b_mouth > cfg.mouth_delta / 2
        ):
            k += 1
        i = k + 1
    return detected


def verify_challenge(
    challenge: Challenge,
    series: dict[str, Any],
    *,
    now: datetime | None = None,
    already_used: bool = False,
    config: LivenessConfig | None = None,
) -> LivenessResult:
    """Check a submitted landmark series against an issued challenge."""
    cfg = config or LivenessConfig()
    now = now or datetime.now(UTC)
    if already_used:
        return LivenessResult(False, 0.0, reasons=["challenge already used"])
    if now > challenge.expires_at:
        return LivenessResult(False, 0.0, reasons=["challenge expired"])
    if series.get("nonce") != challenge.nonce:
        return LivenessResult(False, 0.0, reasons=["series does not belong to this challenge"])
    frames = series.get("frames") or []
    if len(frames) < 5:
        return LivenessResult(False, 0.0, reasons=["too few frames"])
    times = np.asarray([float(f["t"]) for f in frames])
    if np.any(np.diff(times) <= 0):
        return LivenessResult(False, 0.0, reasons=["frame timestamps are not increasing"])
    duration = times[-1] - times[0]
    if duration <= 0 or (len(frames) - 1) / duration < cfg.min_fps:
        return LivenessResult(False, 0.0, reasons=[f"frame rate below {cfg.min_fps} fps"])
    with_face = [f for f in frames if f.get("landmarks") and f.get("faces", 1) == 1]
    if len(with_face) / len(frames) < cfg.min_face_ratio:
        return LivenessResult(False, 0.0, reasons=["exactly one face must stay visible"])
    t = np.asarray([float(f["t"]) for f in with_face])
    lms = [_pts(f["landmarks"]) for f in with_face]
    ear = np.asarray([eye_aspect_ratio(lm) for lm in lms])
    yaw = np.asarray([yaw_proxy(lm) for lm in lms])
    mouth = np.asarray([mouth_aperture(lm) for lm in lms])
    detected = _detect_steps(t, ear, yaw, mouth, challenge.steps, cfg)
    score = len(detected) / len(challenge.steps)
    reasons = [] if score == 1 else [f"did not detect '{challenge.steps[len(detected)]}' in order"]
    return LivenessResult(score == 1.0, score, detected, reasons)
