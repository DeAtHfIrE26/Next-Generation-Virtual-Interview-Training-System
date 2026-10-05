"""Gaze estimation from face-mesh landmarks (E5), upgraded from the prototype.

Prototype: horizontal iris position only, fixed +/-0.20 tolerance
(``interview_core.legacy.gaze``). Here: horizontal and vertical iris position plus head yaw
and pitch, an optional short per-user calibration (look at the centre of the screen for a
few seconds), and temporal smoothing. Output is an observable quantity ("looking away from
the screen 40% of answer 3"), never an inference about feelings or personality.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from interview_core.legacy.gaze import iris_ratios

L_UP, L_LO, R_UP, R_LO = 159, 145, 386, 374
LEFT_IRIS = range(468, 473)
RIGHT_IRIS = range(473, 478)
NOSE, EYE_OUTER = 1, (33, 263)


@dataclass(frozen=True)
class GazeFeatures:
    horizontal: float  # 0 = outer corner, 1 = inner corner, ~0.5 centred
    vertical: float  # 0 = upper lid, 1 = lower lid
    yaw: float  # nose offset / inter-ocular distance
    pitch: float  # nose below eye line / inter-ocular distance

    def as_array(self) -> np.ndarray:
        return np.array([self.horizontal, self.vertical, self.yaw, self.pitch])


def extract_features(landmarks: np.ndarray, width: int = 1, height: int = 1) -> GazeFeatures:
    pts = np.asarray(landmarks, dtype=np.float64)[:, :2] * np.array([width, height], dtype=np.float64)
    lh, rh = iris_ratios(landmarks, width, height)
    li, ri = pts[list(LEFT_IRIS)].mean(0), pts[list(RIGHT_IRIS)].mean(0)
    lv = (li[1] - pts[L_UP][1]) / (pts[L_LO][1] - pts[L_UP][1] + 1e-9)
    rv = (ri[1] - pts[R_UP][1]) / (pts[R_LO][1] - pts[R_UP][1] + 1e-9)
    mid = (pts[EYE_OUTER[0]] + pts[EYE_OUTER[1]]) / 2
    iod = np.linalg.norm(pts[EYE_OUTER[0]] - pts[EYE_OUTER[1]]) + 1e-9
    return GazeFeatures(
        float((lh + rh) / 2),
        float((lv + rv) / 2),
        float((pts[NOSE][0] - mid[0]) / iod),
        float((pts[NOSE][1] - mid[1]) / iod),
    )


@dataclass(frozen=True)
class GazeCalibration:
    mean: np.ndarray
    std: np.ndarray
    k: float = 3.0

    # Minimum spread so a perfectly still calibration does not make every twitch "away".
    FLOOR = np.array([0.04, 0.06, 0.05, 0.05])

    @classmethod
    def fit(cls, samples: Sequence[GazeFeatures], k: float = 3.0) -> GazeCalibration:
        if len(samples) < 15:
            raise ValueError("need >= 15 calibration frames (about 1 s)")
        x = np.stack([s.as_array() for s in samples])
        return cls(x.mean(0), np.maximum(x.std(0), cls.FLOOR), k)


UNCALIBRATED_LIMITS = {"horizontal": 0.20, "vertical": 0.30, "yaw": 0.25}


def is_on_screen(f: GazeFeatures, cal: GazeCalibration | None = None) -> bool:
    if cal is not None:
        return bool(np.all(np.abs((f.as_array() - cal.mean) / cal.std) <= cal.k))
    return (
        abs(f.horizontal - 0.5) <= UNCALIBRATED_LIMITS["horizontal"]
        and abs(f.vertical - 0.5) <= UNCALIBRATED_LIMITS["vertical"]
        and abs(f.yaw) <= UNCALIBRATED_LIMITS["yaw"]
    )


class GazeTracker:
    """Majority vote over the last ``window`` frames to suppress single-frame flicker."""

    def __init__(self, calibration: GazeCalibration | None = None, window: int = 5):
        self.cal = calibration
        self._recent: deque[bool] = deque(maxlen=window)

    def update(self, landmarks: np.ndarray | None, width: int = 1, height: int = 1) -> bool | None:
        if landmarks is None:
            return None
        self._recent.append(is_on_screen(extract_features(landmarks, width, height), self.cal))
        return sum(self._recent) * 2 >= len(self._recent)


def summarise(
    samples: Sequence[tuple[float, bool | None]], min_away_s: float = 0.5
) -> dict[str, float | int]:
    """Observable gaze statistics for one answer from ``(t, on_screen)`` samples."""
    if len(samples) < 2:
        return {"coverage": 0.0, "off_screen_fraction": 0.0, "longest_off_screen_s": 0.0, "look_aways": 0}
    t = np.array([s[0] for s in samples])
    dt = np.diff(t, append=t[-1])
    present = np.array([s[1] is not None for s in samples])
    away = np.array([s[1] is False for s in samples])
    face_time = float(dt[present].sum())
    longest = cur = 0.0
    look_aways = 0
    for d, a in zip(dt, away, strict=True):
        if a:
            cur += d
            if cur >= min_away_s and cur - d < min_away_s:
                look_aways += 1
        else:
            cur = 0.0
        longest = max(longest, cur)
    total = float(t[-1] - t[0]) or 1.0
    return {
        "coverage": round(face_time / total, 3),
        "off_screen_fraction": round(float(dt[away].sum()) / face_time, 3) if face_time else 0.0,
        "longest_off_screen_s": round(longest, 2),
        "look_aways": look_aways,
    }
