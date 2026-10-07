import numpy as np
import pytest
from interview_core.legacy import lipsync as legacy
from interview_core.lipsync import AVSyncConfig, mouth_aperture, verify_av_sync

from .synth import mouth_series, speech_audio, syllable_envelope

SR = 16000


@pytest.fixture
def rng():
    return np.random.default_rng(7)


def _clip(rng, seconds=8.0):
    env = syllable_envelope(rng, seconds)
    return env, speech_audio(env, SR, rng)


@pytest.mark.parametrize("seed", range(5))
def test_genuine_speech_matches_and_offset_is_recovered(seed):
    rng = np.random.default_rng(seed)
    env, audio = _clip(rng)
    t, ap = mouth_series(env, 30, 0.06, rng)
    r = verify_av_sync(audio, SR, t, ap)
    assert r.decision == "match", r
    assert r.offset_ms == pytest.approx(60, abs=25)
    assert r.correlation > 0.5


def test_irregular_low_frame_rate_video_still_matches(rng):
    env, audio = _clip(rng)
    t, ap = mouth_series(env, 15, 0.0, rng, jitter_s=0.02)
    assert verify_av_sync(audio, SR, t, ap).decision == "match"


def test_other_speakers_audio_is_a_mismatch(rng):
    _env, audio = _clip(rng)
    other = syllable_envelope(np.random.default_rng(999), 8.0)
    t, ap = mouth_series(other, 30, 0.0, rng)
    r = verify_av_sync(audio, SR, t, ap)
    assert r.decision == "mismatch"
    assert r.score < 0.3


def test_large_offset_is_flagged(rng):
    env, audio = _clip(rng)
    t, ap = mouth_series(env, 30, 0.4, rng)
    r = verify_av_sync(audio, SR, t, ap)
    assert "audio_video_offset" in r.flags
    assert r.decision == "mismatch"
    assert r.offset_ms == pytest.approx(400, abs=40)


def test_playback_with_still_mouth_is_flagged(rng):
    _env, audio = _clip(rng)
    t = np.arange(0, 8, 1 / 30)
    ap = 0.03 + rng.normal(0, 0.002, t.size)
    r = verify_av_sync(audio, SR, t, ap)
    assert "speech_without_mouth_motion" in r.flags
    assert r.decision == "mismatch" and r.score == 0.0


def test_silence_and_missing_face_are_inconclusive(rng):
    env, audio = _clip(rng)
    t, ap = mouth_series(env, 30, 0.0, rng)
    silent = (0.0005 * rng.normal(0, 1, audio.size)).astype(np.float32)
    assert verify_av_sync(silent, SR, t, ap).decision == "inconclusive"
    assert verify_av_sync(audio, SR, t, np.full(t.size, np.nan)).decision == "inconclusive"
    half = ap.copy()
    half[: t.size // 2 + 30] = np.nan
    assert verify_av_sync(audio, SR, t, half).decision == "inconclusive"


def test_new_check_fires_where_prototype_cannot(rng):
    """Same mismatched clip: the prototype scores >= 0.8 (no warning), the new check flags it."""
    _env, audio = _clip(rng)
    t = np.arange(0, 8, 1 / 30)
    ap = 0.03 + rng.normal(0, 0.002, t.size)
    assert not legacy.would_warn(legacy.verify_lip_sync(float(ap[-1])))
    assert verify_av_sync(audio, SR, t, ap).decision == "mismatch"


def test_thresholds_are_configurable(rng):
    env, audio = _clip(rng)
    t, ap = mouth_series(env, 30, 0.3, rng)
    assert verify_av_sync(audio, SR, t, ap, AVSyncConfig(max_offset_s=0.5)).decision == "match"


def test_mouth_aperture_is_scale_invariant():
    lm = np.random.default_rng(0).uniform(0.2, 0.8, (478, 2))
    big = mouth_aperture(lm, 1280, 960)
    assert big == pytest.approx(mouth_aperture(lm, 640, 480), rel=1e-9)
