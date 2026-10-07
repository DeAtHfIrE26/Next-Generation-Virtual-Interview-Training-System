"""Capture-quality gates for enrolment and verification frames."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class QualityThresholds:
    min_face_fraction: float = 0.12  # face box width / frame width
    min_sharpness: float = 40.0  # variance of Laplacian on the grayscale crop
    min_brightness: float = 50.0
    max_brightness: float = 210.0


@dataclass
class QualityReport:
    ok: bool
    sharpness: float
    brightness: float
    face_fraction: float
    problems: list[str] = field(default_factory=list)


def _laplacian_var(gray: np.ndarray) -> float:
    g = gray.astype(np.float64)
    lap = -4 * g[1:-1, 1:-1] + g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:]
    return float(lap.var())


def assess_quality(
    face_crop_rgb: np.ndarray,
    face_box_width: float,
    frame_width: float,
    num_faces: int,
    t: QualityThresholds | None = None,
) -> QualityReport:
    t = t or QualityThresholds()
    gray = (
        face_crop_rgb.astype(np.float64) @ np.array([0.299, 0.587, 0.114])
        if face_crop_rgb.ndim == 3
        else face_crop_rgb
    )
    sharp, bright = _laplacian_var(gray), float(gray.mean())
    frac = face_box_width / frame_width if frame_width else 0.0
    problems = []
    if num_faces != 1:
        problems.append("exactly one face must be visible" if num_faces else "no face found")
    if frac < t.min_face_fraction:
        problems.append("move closer to the camera")
    if sharp < t.min_sharpness:
        problems.append("image is blurry; hold still")
    if bright < t.min_brightness:
        problems.append("too dark; add light in front of you")
    if bright > t.max_brightness:
        problems.append("too bright; avoid light directly behind or on the camera")
    return QualityReport(not problems, sharp, bright, frac, problems)
