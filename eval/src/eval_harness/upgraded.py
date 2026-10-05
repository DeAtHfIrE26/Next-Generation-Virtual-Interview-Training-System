"""Registration of upgraded (post-prototype) systems for side-by-side evaluation."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np
from interview_core.adapters import factory
from interview_core.biometric import build_template, cosine
from interview_core.face.liveness import Challenge, verify_challenge
from interview_core.gaze import extract_features, is_on_screen
from interview_core.lipsync import mouth_aperture, verify_av_sync


def _lipsync_v2(audio: np.ndarray, sr: int, series: dict[str, Any]) -> float:
    fps = float(series.get("fps", 30))
    if series.get("times"):
        times = np.asarray(series["times"], float)
    else:
        n = len(series.get("landmarks") or series.get("aperture") or [])
        times = np.arange(n) / fps
    if series.get("landmarks"):
        values = np.array([mouth_aperture(np.asarray(lm)) if lm else np.nan for lm in series["landmarks"]])
    else:
        values = np.asarray([np.nan if v is None else v for v in series["aperture"]], float)
    r = verify_av_sync(audio, sr, times, values)
    # Inconclusive clips carry no evidence either way.
    return 0.5 if r.decision == "inconclusive" else r.score


def _gaze_v2(landmarks: np.ndarray | None, width: int, height: int) -> bool | None:
    if landmarks is None:
        return None
    return is_on_screen(extract_features(landmarks, width, height))


def _liveness_active(series: dict[str, Any]) -> float:
    c = series["challenge"]
    ch = Challenge(
        c["nonce"],
        tuple(c["steps"]),
        datetime.fromisoformat(c["issued_at"]),
        datetime.fromisoformat(c["expires_at"]),
    )
    return verify_challenge(ch, series, now=ch.issued_at).score


def register(face, lipsync, gaze, answer, speaker, asr, liveness) -> list[str]:
    notes: list[str] = []
    lipsync["avsync_v2"] = _lipsync_v2
    gaze["gaze_v2_uncalibrated"] = _gaze_v2
    liveness["active_challenge_v1"] = _liveness_active

    emb = factory.face_embedder()
    if emb is not None:

        def score_face(enrol, probe, _emb=emb):
            rgb = lambda g: np.repeat(g[..., None], 3, axis=2) if g.ndim == 2 else g  # noqa: E731
            tpl = build_template("face", _emb.model_id, [_emb.embed(rgb(e)) for e in enrol], min_samples=1)
            return cosine(tpl.vector, _emb.embed(rgb(probe)))

        face[emb.model_id] = score_face
    else:
        notes.append("face embedder not configured (FACE_EMBEDDER); only legacy_histogram evaluated")

    spk = factory.speaker_embedder()
    if spk is not None:

        def score_voice(enrol, probe, sr, _spk=spk):
            tpl = build_template("voice", _spk.model_id, [_spk.embed(a, sr) for a in enrol], min_samples=1)
            return cosine(tpl.vector, _spk.embed(probe, sr))

        speaker[spk.model_id] = score_voice
    else:
        notes.append("speaker embedder not configured (SPEAKER_EMBEDDER)")

    try:
        from interview_core.nlp.registry import eval_answer_scorers, eval_asr_systems

        answer.update(eval_answer_scorers())
        asr.update(eval_asr_systems())
    except ImportError:
        pass
    return notes
