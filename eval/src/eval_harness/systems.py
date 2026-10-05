"""Registry of algorithm versions ("systems") that suites compare.

Each suite reports every available system side by side, which is how before/after numbers
are produced: ``legacy_*`` systems are the prototype algorithms from
``interview_core.legacy``; the others are the upgraded implementations. Systems that need
a model file or vendor credentials appear only when configured (see ``.env.example``).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from interview_core.legacy import face_hist as legacy_face
from interview_core.legacy import gaze as legacy_gaze
from interview_core.legacy import grading as legacy_grading
from interview_core.legacy import lipsync as legacy_lipsync

# Face: (enrolment images, probe image) -> similarity, higher = same person
FaceScorer = Callable[[list[np.ndarray], np.ndarray], float]
# Lip-sync: (audio, sample_rate, mouth_series dict) -> authenticity score, higher = genuine
LipSyncScorer = Callable[[np.ndarray, int, dict[str, Any]], float]
# Gaze: (landmarks, width, height) -> True on-screen / False away / None no face
GazePredictor = Callable[[np.ndarray | None, int, int], bool | None]
# Answer scoring: (question, answer, role, context) -> score in [0, 1]
AnswerScorer = Callable[[str, str, str, str], float]


def _legacy_face(enrol: list[np.ndarray], probe: np.ndarray) -> float:
    return max(legacy_face.compare_face_hist(probe, ref) for ref in enrol)


def _legacy_lipsync(_audio: np.ndarray, _sr: int, series: dict[str, Any]) -> float:
    """The prototype scored one frame: the latest one when the utterance ended."""
    if series.get("landmarks"):
        lm = np.asarray(series["landmarks"][-1])
        ratio = legacy_lipsync.mouth_opening_ratio(lm, series["width"], series["height"])
    elif series.get("aperture"):
        ratio = float(series["aperture"][-1])
    else:
        ratio = None
    return legacy_lipsync.verify_lip_sync(ratio)


def _legacy_answer(_q: str, answer: str, role: str, context: str) -> float:
    _, breakdown, _ = legacy_grading.grade_interview_with_breakdown(
        f"Candidate: {answer}", context, job_role=role
    )
    details = breakdown.get("per_response_details") or [{"response_score": 0.0}]
    return float(details[0]["response_score"]) / 100.0


FACE: dict[str, FaceScorer] = {"legacy_histogram": _legacy_face}
LIPSYNC: dict[str, LipSyncScorer] = {"legacy_single_frame": _legacy_lipsync}
GAZE: dict[str, GazePredictor] = {"legacy_iris_horizontal": legacy_gaze.detect_eye_gaze}
ANSWER: dict[str, AnswerScorer] = {"legacy_keyword_heuristic": _legacy_answer}
SPEAKER: dict[str, Callable[[list[np.ndarray], np.ndarray, int], float]] = {}
ASR: dict[str, Callable[[np.ndarray, int], str]] = {}
LIVENESS: dict[str, Callable[[dict[str, Any]], float]] = {}


def register_optional() -> list[str]:
    """Register systems implemented in later modules. Returns notes about what was skipped."""
    notes: list[str] = []
    try:
        from eval_harness import upgraded

        notes.extend(upgraded.register(FACE, LIPSYNC, GAZE, ANSWER, SPEAKER, ASR, LIVENESS))
    except ImportError as e:  # pragma: no cover - only before upgraded systems exist
        notes.append(f"upgraded systems unavailable: {e}")
    return notes
