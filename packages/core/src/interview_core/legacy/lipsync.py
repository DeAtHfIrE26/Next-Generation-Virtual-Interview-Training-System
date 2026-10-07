"""Prototype lip-sync verification (E4).

Source: ``legacy/desktop/main.py`` ``compute_mouth_opening`` (L1002-1018) and
``verify_lip_sync`` (L948-974).

Known defect preserved: :func:`verify_lip_sync` floors its score at 0.8 and the caller
warns only when the score is below 0.8, so the check can never raise a warning. It also
looks at a single video frame and ignores the audio. :mod:`interview_core.lipsync.avsync`
is the working replacement.
"""

from __future__ import annotations

import numpy as np

MIN_MOUTH_MOVEMENT_RATIO = 0.02
WARN_BELOW = 0.8

UPPER_LIP, LOWER_LIP, FACE_TOP, FACE_BOTTOM = 13, 14, 10, 152


def mouth_opening_ratio(landmarks: np.ndarray, width: int, height: int) -> float:
    """Inner-lip gap over face height, from normalised MediaPipe face-mesh landmarks."""
    pts = np.asarray(landmarks, dtype=np.float64)[:, :2] * np.array([width, height], dtype=np.float64)
    opening = np.linalg.norm(pts[UPPER_LIP] - pts[LOWER_LIP])
    face_h = np.linalg.norm(pts[FACE_TOP] - pts[FACE_BOTTOM])
    return float(opening / (face_h + 1e-6))


def verify_lip_sync(ratio: float | None) -> float:
    if ratio is None:
        return 1.0
    min_threshold = MIN_MOUTH_MOVEMENT_RATIO * 0.5
    return max(0.8, min(1.0, ratio / min_threshold))


def would_warn(score: float) -> bool:
    return score < WARN_BELOW
