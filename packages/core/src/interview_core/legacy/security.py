"""Prototype security monitoring rules (E7).

Source: ``legacy/desktop/main.py`` thresholds (L220-254), ``check_same_person_and_phone``
(L1286-1348) and ``monitor_background_audio`` (L1020-1054).
"""

from __future__ import annotations

import numpy as np

PHONE_LABELS = frozenset({"cell phone", "mobile phone", "phone"})
PHONE_DETECT_THRESHOLD = 3
BACKGROUND_NOISE_RMS = 3000.0


def phone_counter_step(counter: int, phone_in_frame: bool) -> tuple[int, bool]:
    """Return ``(new_counter, raise_phone_warning)``. Needs N consecutive detections."""
    if phone_in_frame:
        counter += 1
        return counter, counter >= PHONE_DETECT_THRESHOLD
    return 0, False


def is_phone_label(label: str) -> bool:
    return label.lower() in PHONE_LABELS


def background_noise_exceeded(pcm_int16: np.ndarray) -> bool:
    energy = float(np.sqrt(np.mean(np.asarray(pcm_int16).astype(np.float32) ** 2)))
    return energy > BACKGROUND_NOISE_RMS
