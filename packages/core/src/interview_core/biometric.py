"""Shared embedding-template logic for face (E2) and voice (E3) verification."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal

import numpy as np


class NotCalibratedError(RuntimeError):
    """Raised when a decision is requested but no calibrated threshold is configured."""


@dataclass(frozen=True)
class Template:
    """Enrolment template: unit-norm mean embedding plus provenance."""

    kind: Literal["face", "voice"]
    model_id: str
    vector: np.ndarray
    n_samples: int
    created_at: datetime

    def to_bytes(self) -> bytes:
        return np.asarray(self.vector, dtype="<f4").tobytes()

    @classmethod
    def from_bytes(cls, kind, model_id: str, raw: bytes, n_samples: int, created_at: datetime) -> Template:
        return cls(kind, model_id, np.frombuffer(raw, dtype="<f4").copy(), n_samples, created_at)


@dataclass(frozen=True)
class VerifyResult:
    score: float  # cosine similarity in [-1, 1]
    accepted: bool | None  # None when no calibrated threshold is configured
    threshold: float | None
    model_id: str


def l2_normalise(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64).ravel()
    n = np.linalg.norm(v)
    if n < 1e-12:
        raise ValueError("zero-length embedding")
    return v / n


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(l2_normalise(a), l2_normalise(b)))


def build_template(
    kind: Literal["face", "voice"],
    model_id: str,
    embeddings: Sequence[np.ndarray],
    *,
    min_samples: int,
    outlier_cosine: float = 0.5,
) -> Template:
    """Average unit-norm embeddings after dropping samples far from the median direction.

    Outliers (a different person stepping in, a cough, a blurred frame) would otherwise pull
    the template off. At least ``min_samples`` inliers are required.
    """
    if len(embeddings) < min_samples:
        raise ValueError(f"need at least {min_samples} samples, got {len(embeddings)}")
    unit = np.stack([l2_normalise(e) for e in embeddings])
    median_dir = l2_normalise(np.median(unit, axis=0))
    keep = unit @ median_dir >= outlier_cosine
    if keep.sum() < min_samples:
        raise ValueError(
            f"only {int(keep.sum())} consistent samples (need {min_samples}); recapture enrolment"
        )
    vec = l2_normalise(unit[keep].mean(axis=0)).astype(np.float32)
    return Template(kind, model_id, vec, int(keep.sum()), datetime.now(UTC))


def verify(template: Template, probe: np.ndarray, threshold: float | None, model_id: str) -> VerifyResult:
    if model_id != template.model_id:
        raise ValueError(f"template made with {template.model_id}, probe with {model_id}; re-enrol")
    score = cosine(template.vector, probe)
    accepted = None if threshold is None else score >= threshold
    return VerifyResult(score, accepted, threshold, model_id)
