"""Prototype voice matching (E3).

Source: ``legacy/desktop/main.py`` ``cosine_similarity`` (L940-944) and ``record_audio``
(L2401-2472). Known defect preserved in :func:`answer_voice_check`: when no reference
exists, the first answer silently becomes the reference (enrolment bypass).
"""

from __future__ import annotations

import numpy as np

VOICE_MATCH_THRESHOLD = 0.8
END_SESSION_WARNINGS = 3


def cosine_similarity(a: np.ndarray | None, b: np.ndarray | None) -> float:
    if a is None or b is None:
        return 0
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-6))


def answer_voice_check(
    reference: np.ndarray | None, current: np.ndarray, warning_count: int
) -> tuple[np.ndarray, int, bool]:
    """Return ``(reference, warning_count, end_session)`` after one recorded utterance."""
    if reference is None:
        return current, warning_count, False  # defect: enrolment bypass
    if cosine_similarity(reference, current) < VOICE_MATCH_THRESHOLD:
        warning_count += 1
        return reference, warning_count, warning_count >= END_SESSION_WARNINGS
    return reference, warning_count, False
