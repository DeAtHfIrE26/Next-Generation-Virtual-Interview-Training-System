"""Characterization tests: interview_core.legacy must match the prototype exactly.

Each test runs the original prototype function (parsed from legacy/desktop/main.py) and
our re-implementation on the same inputs. These pin behaviour before any upgrade.
"""

from __future__ import annotations

import re
import traceback
from types import SimpleNamespace

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from interview_core.legacy import face_hist, gaze, grading, lipsync, resume, voice

from .legacy_loader import load_functions

cv2 = pytest.importorskip("cv2")

VOCAB = [
    "um",
    "uh",
    "like",
    "basically",
    "you know",
    "i mean",
    "summary",
    "team",
    "we",
    "our",
    "together",
    "first",
    "then",
    "finally",
    "because",
    "however",
    "for example",
    "specifically",
    "algorithm",
    "database",
    "api",
    "cloud",
    "code",
    "function",
    "class",
    "big-o",
    "edge case",
    "time complexity",
    "maybe",
    "i think",
    "definitely",
    "i am confident",
    "approach",
    "solution",
    "design",
    "debug",
    "test",
    "problem",
    "result",
    "python",
    "customer",
    "project",
    "deadline",
    "collaborated on",
    "cross-functional",
    "identify",
    "analyze",
    "solve",
    "the",
    "a",
    "and",
]
ROLES = [None, "", "Software Engineer", "Data Analyst", "Product Manager", "HR Generalist", "DevOps SDE"]


def _legacy_grader(job_role, eye_away, total_checks, warnings):
    ns = {
        "re": re,
        "traceback": traceback,
        "log_event": lambda *_a, **_k: None,
        "job_role": job_role,
        "warning_count": warnings,
    }
    if total_checks is not None:
        ns["eye_away_count"] = eye_away
        ns["total_eye_checks"] = total_checks
    else:
        # The prototype checks `"eye_away_count" not in globals()`; emulate "never set".
        ns["eye_away_count"] = 0
        ns["total_eye_checks"] = 0
    return load_functions(["grade_interview_with_breakdown"], ns)["grade_interview_with_breakdown"]


answers = st.lists(st.sampled_from(VOCAB), min_size=0, max_size=120).map(" ".join)


@settings(max_examples=300, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    responses=st.lists(answers, min_size=0, max_size=6),
    context=st.lists(st.sampled_from(VOCAB), max_size=40).map(" ".join),
    role=st.sampled_from(ROLES),
    eye=st.tuples(st.integers(0, 50), st.integers(0, 50)),
    warnings=st.integers(0, 12),
)
def test_grading_matches_prototype(responses, context, role, eye, warnings):
    away, total = eye
    away = min(away, total)
    transcript = "\n".join(["Interviewer: Tell me about yourself."] + [f"Candidate: {r}" for r in responses])
    legacy = _legacy_grader(role, away, total, warnings)(transcript, context)
    ours = grading.grade_interview_with_breakdown(
        transcript,
        context,
        job_role=role,
        eye_away_count=away,
        total_eye_checks=total,
        warning_count=warnings,
    )
    assert ours == legacy


@settings(max_examples=50, deadline=None)
@given(seed=st.integers(0, 10_000), mode=st.sampled_from(["random", "flat", "same"]))
def test_face_histogram_correlation_matches_opencv(seed, mode):
    rng = np.random.default_rng(seed)
    a = rng.integers(0, 256, (200, 200), dtype=np.uint8)
    if mode == "flat":
        b = np.full((200, 200), rng.integers(0, 256), dtype=np.uint8)
    elif mode == "same":
        b = a.copy()
    else:
        b = rng.integers(0, 256, (200, 200), dtype=np.uint8)
    ns = {"cv2": cv2}
    legacy = load_functions(["compare_face_hist"], ns)["compare_face_hist"](a, b)
    assert face_hist.compare_face_hist(a, b) == pytest.approx(legacy, abs=1e-5)


def test_face_decision_rules():
    assert face_hist.same_person_decision(0, 0)[0] is False
    assert face_hist.same_person_decision(1, 1) == (True, "OK")
    assert "Another unauthorized person" in face_hist.same_person_decision(2, 1)[1]
    assert "mismatch" in face_hist.same_person_decision(1, 0)[1]
    with pytest.raises(ValueError):
        face_hist.validate_samples([np.zeros((10, 10))] * 14)


def test_histogram_matcher_cannot_separate_identities():
    """Documents why E2 needed an upgrade: a shifted-content image with the same
    intensity distribution is a 'perfect' match."""
    rng = np.random.default_rng(0)
    a = rng.integers(0, 256, (200, 200), dtype=np.uint8)
    b = np.roll(a, 97, axis=1)  # different spatial content, identical histogram
    assert face_hist.compare_face_hist(a, b) == pytest.approx(1.0)


@given(
    a=st.lists(st.floats(-5, 5, allow_nan=False), min_size=8, max_size=8),
    b=st.lists(st.floats(-5, 5, allow_nan=False), min_size=8, max_size=8),
)
def test_voice_cosine_matches_prototype(a, b):
    legacy = load_functions(["cosine_similarity"], {"np": np})["cosine_similarity"]
    va, vb = np.array(a), np.array(b)
    assert voice.cosine_similarity(va, vb) == pytest.approx(legacy(va, vb), rel=1e-9, abs=1e-12)
    assert voice.cosine_similarity(None, vb) == legacy(None, vb) == 0


def test_voice_enrolment_bypass_defect_is_pinned():
    first = np.ones(4)
    ref, warnings, end = voice.answer_voice_check(None, first, 0)
    assert ref is first and warnings == 0 and not end  # defect: answer became the reference


class _FakeMesh:
    """Stand-in for mediapipe FaceMesh.process(); test-only."""

    def __init__(self, landmarks):
        self._lm = landmarks

    def process(self, _rgb):
        if self._lm is None:
            return SimpleNamespace(multi_face_landmarks=None)
        pts = [SimpleNamespace(x=float(x), y=float(y)) for x, y in self._lm]
        return SimpleNamespace(multi_face_landmarks=[SimpleNamespace(landmark=pts)])


def _random_landmarks(rng):
    return rng.uniform(0.05, 0.95, size=(478, 2))


@settings(max_examples=100, deadline=None)
@given(seed=st.integers(0, 100_000))
def test_mouth_ratio_and_lipsync_match_prototype(seed):
    rng = np.random.default_rng(seed)
    lm = _random_landmarks(rng)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    mesh = _FakeMesh(lm)
    ns = {
        "cv2": cv2,
        "np": np,
        "mp_face_mesh": mesh,
        "MIN_MOUTH_MOVEMENT_RATIO": 0.02,
        "log_event": lambda *_a: None,
    }
    fns = load_functions(["compute_mouth_opening", "verify_lip_sync"], ns)
    legacy_ratio = fns["compute_mouth_opening"](frame, mesh)
    ours_ratio = lipsync.mouth_opening_ratio(lm, 640, 480)
    assert ours_ratio == pytest.approx(legacy_ratio, rel=1e-9)
    assert lipsync.verify_lip_sync(ours_ratio) == pytest.approx(fns["verify_lip_sync"](b"", frame))


@given(ratio=st.one_of(st.none(), st.floats(0, 1, allow_nan=False)))
def test_legacy_lipsync_can_never_warn(ratio):
    """Pins the defect recorded in docs/CLAIM_MAP.md (E4): the score floor equals the
    warning threshold, so the prototype check never fires."""
    assert not lipsync.would_warn(lipsync.verify_lip_sync(ratio))


@settings(max_examples=100, deadline=None)
@given(seed=st.integers(0, 100_000), no_face=st.booleans())
def test_gaze_matches_prototype(seed, no_face):
    rng = np.random.default_rng(seed)
    lm = None if no_face else _random_landmarks(rng)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    legacy = load_functions(["detect_eye_gaze"], {"cv2": cv2, "np": np})["detect_eye_gaze"]
    assert gaze.detect_eye_gaze(lm, 640, 480) == legacy(frame, _FakeMesh(lm))


@given(text=st.text(alphabet=st.sampled_from(list("ab Z\n9-")), max_size=900), role=st.text(max_size=20))
def test_resume_context_matches_prototype(text, role):
    fns = load_functions(["extract_candidate_name", "build_context"], {"re": re})
    assert resume.extract_candidate_name(text) == fns["extract_candidate_name"](text)
    assert resume.build_context(text, role) == fns["build_context"](text, role)
