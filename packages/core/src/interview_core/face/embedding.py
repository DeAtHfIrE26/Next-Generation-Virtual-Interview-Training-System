"""Embedding-based face verification.

Replaces the prototype's intensity-histogram correlation, which cannot distinguish people
(see ``interview_core.legacy.face_hist``), while keeping the claimed flow: capture multiple
facial samples, build a reference, verify the candidate against it during the session.

The embedding model is pluggable. Only commercially licensed models may be configured in
production (see MODEL_CARDS/face.md). The threshold must come from evaluation calibration.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import numpy as np

from interview_core.biometric import Template, VerifyResult, build_template, verify

MIN_ENROL_SAMPLES = 5


class FaceEmbedder(Protocol):
    model_id: str

    def embed(self, face_rgb: np.ndarray) -> np.ndarray:
        """Return an embedding for one aligned face crop (H x W x 3, uint8, RGB)."""
        ...


class FaceVerifier:
    def __init__(self, embedder: FaceEmbedder, threshold: float | None):
        self.embedder = embedder
        self.threshold = threshold

    def enrol(self, faces: Sequence[np.ndarray]) -> Template:
        embs = [self.embedder.embed(f) for f in faces]
        return build_template("face", self.embedder.model_id, embs, min_samples=MIN_ENROL_SAMPLES)

    def verify(self, template: Template, face: np.ndarray) -> VerifyResult:
        return verify(template, self.embedder.embed(face), self.threshold, self.embedder.model_id)
