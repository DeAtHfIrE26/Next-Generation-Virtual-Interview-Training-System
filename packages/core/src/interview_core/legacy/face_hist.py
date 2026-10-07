"""Prototype face "recognizer" (E2): grayscale intensity-histogram correlation.

Source: ``legacy/desktop/main.py`` ``compare_face_hist``/``predict_face``/``train_lbph``
(L1173-1220) and the decision in ``check_same_person_and_phone`` (L1286-1348).

Despite the prototype's "LBPH" naming, this compares 256-bin intensity histograms with
OpenCV's correlation metric. A histogram discards all spatial structure, so two different
people under similar lighting can score near 1.0. It is kept only as the regression
reference; :mod:`interview_core.face` replaces it with embedding verification.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

LBPH_THRESHOLD = 70
REGISTERED_LABEL = 100
MIN_SAMPLES = 15


def _minmax_hist(gray: np.ndarray) -> np.ndarray:
    hist = np.bincount(np.asarray(gray, dtype=np.uint8).ravel(), minlength=256).astype(np.float32)
    lo, hi = float(hist.min()), float(hist.max())
    if hi - lo == 0:
        return np.zeros_like(hist)
    return ((hist - lo) / (hi - lo)).astype(np.float32)


def compare_face_hist(face_a: np.ndarray, face_b: np.ndarray) -> float:
    """Equivalent of ``cv2.compareHist(..., HISTCMP_CORREL)`` on MINMAX-normalised histograms."""
    h1 = _minmax_hist(face_a).astype(np.float64)
    h2 = _minmax_hist(face_b).astype(np.float64)
    d1, d2 = h1 - h1.mean(), h2 - h2.mean()
    denom = np.sqrt((d1 * d1).sum() * (d2 * d2).sum())
    if denom == 0:
        return 1.0 if np.array_equal(h1, h2) else 0.0
    return float((d1 * d2).sum() / denom)


def validate_samples(samples: Sequence[np.ndarray]) -> None:
    if len(samples) < MIN_SAMPLES:
        raise ValueError("Not enough valid face samples captured. Keep face visible longer.")


def predict_face(face: np.ndarray, reference_faces: Sequence[np.ndarray] | None) -> tuple[int, int]:
    """Return ``(label, confidence)``; lower confidence means a better match, like LBPH."""
    if not reference_faces:
        return -1, 0
    best = max(compare_face_hist(face, ref) for ref in reference_faces)
    return REGISTERED_LABEL, int((1 - best) * 100)


def same_person_decision(num_faces: int, recognized_count: int) -> tuple[bool, str]:
    """Outcome rule of ``check_same_person_and_phone`` once faces are detected and matched."""
    if num_faces == 0:
        return False, "No face detected."
    if num_faces == 1 and recognized_count == 1:
        return True, "OK"
    if num_faces > 1:
        return False, "Another unauthorized person is in the frame. Only the registered user is allowed."
    return False, "Face mismatch detected. Please ensure your face is properly registered."
