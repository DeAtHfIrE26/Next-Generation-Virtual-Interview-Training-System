"""Mouth-opening measurement from MediaPipe face-mesh landmarks (478-point topology)."""

from __future__ import annotations

import numpy as np

# Inner-lip vertical pairs (upper, lower): centre, left-centre, right-centre.
INNER_LIP_PAIRS = ((13, 14), (82, 87), (312, 317))
# Outer eye corners: the inter-ocular distance normalises for camera distance.
EYE_OUTER = (33, 263)


def mouth_aperture(landmarks: np.ndarray, width: int = 1, height: int = 1) -> float:
    """Mean inner-lip gap divided by inter-ocular distance (scale-invariant, roughly 0-0.6).

    Using three lip pairs instead of one makes the value robust to asymmetric speech and
    landmark jitter. Normalising by eye distance instead of face height makes it insensitive
    to jaw opening changing the denominator.
    """
    pts = np.asarray(landmarks, dtype=np.float64)[:, :2] * np.array([width, height], dtype=np.float64)
    gap = np.mean([np.linalg.norm(pts[u] - pts[lo]) for u, lo in INNER_LIP_PAIRS])
    iod = np.linalg.norm(pts[EYE_OUTER[0]] - pts[EYE_OUTER[1]])
    return float(gap / (iod + 1e-9))
