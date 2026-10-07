"""Prototype eye tracking (E5).

Source: ``legacy/desktop/main.py`` ``detect_eye_gaze`` (L976-1000): horizontal iris
position relative to the eye corners, with a fixed +/-0.20 tolerance around centre.
"""

from __future__ import annotations

import numpy as np

LEFT_IRIS = range(468, 473)
RIGHT_IRIS = range(473, 478)
LEFT_CORNERS = (33, 133)
RIGHT_CORNERS = (263, 362)
THRESHOLD = 0.20


def iris_ratios(landmarks: np.ndarray, width: int, height: int) -> tuple[float, float]:
    pts = np.asarray(landmarks, dtype=np.float64)[:, :2] * np.array([width, height], dtype=np.float64)
    li = pts[list(LEFT_IRIS)].mean(axis=0)
    ri = pts[list(RIGHT_IRIS)].mean(axis=0)
    l0, l1 = pts[LEFT_CORNERS[0]], pts[LEFT_CORNERS[1]]
    r0, r1 = pts[RIGHT_CORNERS[0]], pts[RIGHT_CORNERS[1]]
    left = (li[0] - l0[0]) / (l1[0] - l0[0] + 1e-6)
    right = (ri[0] - r0[0]) / (r1[0] - r0[0] + 1e-6)
    return float(left), float(right)


def detect_eye_gaze(landmarks: np.ndarray | None, width: int, height: int) -> bool | None:
    """True = looking at screen, False = away, None = no face."""
    if landmarks is None:
        return None
    left, right = iris_ratios(landmarks, width, height)
    return not (abs(left - 0.5) > THRESHOLD or abs(right - 0.5) > THRESHOLD)
