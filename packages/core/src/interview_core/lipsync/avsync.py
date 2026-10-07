"""Temporal audio-visual synchronisation check (E4, stage 1).

The patent element is "a lip-sync verification module for analyzing mouth movements and
detecting speech authenticity mismatches". The prototype looked at one frame and could not
fire (see ``interview_core.legacy.lipsync``). This module analyses the mouth-opening time
series against the speech energy envelope over the whole utterance:

1. speech energy envelope (300-3000 Hz band, 100 Hz frame rate, log-compressed) and an
   adaptive energy voice-activity mask;
2. mouth-aperture series (from face landmarks, any frame rate, gaps allowed) resampled onto
   the same 50 Hz grid;
3. normalised cross-correlation over lags of +/- ``max_search_lag_s``; the peak gives a
   correlation strength and the audio-to-video offset;
4. flags for speech with a still mouth (playback / someone else speaking), an offset outside
   tolerance (dubbed or delayed audio), and sustained mouth motion without speech.

All thresholds are provisional defaults and must be calibrated on the ``lipsync`` evaluation
suite before decisions are enforced (see eval/DATA_COLLECTION.md).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from interview_core import signals

Decision = Literal["match", "mismatch", "inconclusive"]


@dataclass(frozen=True)
class AVSyncConfig:
    grid_hz: float = 50.0
    max_search_lag_s: float = 0.5
    max_offset_s: float = 0.2  # typical A/V sync tolerance; larger offsets are flagged
    min_correlation: float = 0.3
    # Peak correlation must stand out from correlations at implausible (circular) shifts of
    # the same signals; this guards against chance alignment of any two syllable trains.
    min_prominence: float = 3.0
    null_min_shift_s: float = 1.0
    min_voiced_s: float = 1.0
    min_mouth_coverage: float = 0.6  # share of voiced time with a usable face
    min_voiced_mouth_std: float = 0.02  # aperture units (inter-ocular normalised)
    silent_motion_s: float = 1.5  # sustained mouth motion with no speech
    band_hz: tuple[float, float] = (300.0, 3000.0)


@dataclass
class AVSyncResult:
    decision: Decision
    score: float  # 0..1, higher = more consistent with genuine speech
    correlation: float | None
    offset_ms: float | None  # positive = mouth moves after the audio
    voiced_s: float
    mouth_coverage: float
    flags: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    prominence: float | None = None


def _prominence(a: np.ndarray, b: np.ndarray, peak: float, min_shift: int) -> float | None:
    """Robust z-score of ``peak`` against correlations of ``a`` with circular shifts of ``b``."""
    n = min(a.size, b.size)
    shifts = range(min_shift, n - min_shift + 1, max(1, (n - 2 * min_shift) // 60 or 1))
    null = []
    for s in shifts:
        bs = np.roll(b[:n], s)
        sa, sb = a[:n].std(), bs.std()
        if sa > 1e-12 and sb > 1e-12:
            null.append(float(np.mean((a[:n] - a[:n].mean()) * (bs - bs.mean())) / (sa * sb)))
    if len(null) < 10:
        return None
    null_arr = np.asarray(null)
    med = np.median(null_arr)
    mad = np.median(np.abs(null_arr - med)) * 1.4826
    return float((peak - med) / max(mad, 0.02))


def _voice_activity(env: np.ndarray) -> np.ndarray:
    if env.size == 0:
        return np.zeros(0, dtype=bool)
    floor = np.percentile(env, 10)
    peak = np.percentile(env, 95)
    if peak < 1e-4 or peak < 3 * max(floor, 1e-9):
        return np.zeros(env.size, dtype=bool)
    return env > floor + 0.25 * (peak - floor)


def _longest_run(mask: np.ndarray) -> int:
    best = cur = 0
    for v in mask:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return best


def verify_av_sync(
    audio: np.ndarray,
    sr: int,
    mouth_times_s: np.ndarray,
    mouth_values: np.ndarray,
    config: AVSyncConfig | None = None,
) -> AVSyncResult:
    """Check one utterance.

    ``mouth_times_s`` are seconds relative to the first audio sample; ``mouth_values`` are
    apertures (``nan`` where no face was found).
    """
    cfg = config or AVSyncConfig()
    audio = np.asarray(audio, dtype=np.float32)
    t_m = np.asarray(mouth_times_s, dtype=np.float64)
    v_m = np.asarray(mouth_values, dtype=np.float64)

    band = signals.bandpass_fft(audio, sr, *cfg.band_hz)
    env100 = signals.rms_envelope(band, sr, rate_hz=100.0)
    if env100.size < 10:
        return AVSyncResult("inconclusive", 0.0, None, None, 0.0, 0.0, reasons=["utterance too short"])
    t_env = (np.arange(env100.size) * (sr / 100.0) + 0.0125 * sr) / sr
    voiced100 = _voice_activity(env100)
    voiced_s = float(voiced100.sum() / 100.0)

    ok = np.isfinite(v_m)
    if ok.sum() < 3:
        return AVSyncResult("inconclusive", 0.0, None, None, voiced_s, 0.0, reasons=["no usable face frames"])

    # Coverage is measured over the whole utterance: speech heard while no face was visible
    # counts against it, so a partly hidden face cannot be "verified" on the visible part only.
    full_grid = np.arange(t_env[0], t_env[-1], 1 / cfg.grid_hz)
    voiced_full = signals.resample_series(t_env, voiced100.astype(float), full_grid) > 0.5
    face_t = np.sort(t_m[ok])
    pos = np.clip(np.searchsorted(face_t, full_grid), 1, face_t.size - 1)
    nearest = np.minimum(np.abs(full_grid - face_t[pos - 1]), np.abs(full_grid - face_t[pos]))
    has_face = nearest <= 0.1
    voiced_pts = int(voiced_full.sum())
    coverage = float((has_face & voiced_full).sum() / voiced_pts) if voiced_pts else 0.0
    flags: list[str] = []
    reasons: list[str] = []

    if voiced_s < cfg.min_voiced_s:
        return AVSyncResult(
            "inconclusive",
            0.0,
            None,
            None,
            voiced_s,
            coverage,
            reasons=[f"only {voiced_s:.1f}s of speech (< {cfg.min_voiced_s}s)"],
        )
    if coverage < cfg.min_mouth_coverage:
        return AVSyncResult(
            "inconclusive",
            0.0,
            None,
            None,
            voiced_s,
            coverage,
            reasons=[f"face visible for {coverage:.0%} of speech"],
        )

    t0 = max(t_env[0], face_t.min())
    t1 = min(t_env[-1], face_t.max())
    grid = np.arange(t0, t1, 1 / cfg.grid_hz)
    log_env = np.log(env100 + 1e-4)
    env_g = signals.resample_series(t_env, signals.moving_average(log_env, 5), grid)
    voiced_g = signals.resample_series(t_env, voiced100.astype(float), grid) > 0.5
    mouth_g = signals.resample_series(t_m[ok], signals.moving_average(v_m[ok], 3), grid)

    max_lag = round(cfg.max_search_lag_s * cfg.grid_hz)
    za, zb = signals.zscore(env_g), signals.zscore(mouth_g)
    lags, corr = signals.lagged_correlation(za, zb, max_lag)
    if np.all(np.isnan(corr)):
        return AVSyncResult(
            "inconclusive", 0.0, None, None, voiced_s, coverage, reasons=["correlation undefined"]
        )
    best = int(np.nanargmax(corr))
    r = float(corr[best])
    offset_s = float(lags[best] / cfg.grid_hz)

    prominence = _prominence(za, np.roll(zb, -lags[best]), r, int(cfg.null_min_shift_s * cfg.grid_hz))
    voiced_mouth_std = float(mouth_g[voiced_g].std()) if voiced_g.any() else 0.0
    if voiced_mouth_std < cfg.min_voiced_mouth_std:
        flags.append("speech_without_mouth_motion")
        reasons.append(f"mouth barely moved while speech was heard (std {voiced_mouth_std:.3f})")
    if abs(offset_s) > cfg.max_offset_s:
        flags.append("audio_video_offset")
        reasons.append(
            f"lips {'trail' if offset_s > 0 else 'lead'} the audio by {abs(offset_s) * 1000:.0f} ms"
        )
    unvoiced_motion = (~voiced_g) & (np.abs(np.gradient(mouth_g)) * cfg.grid_hz > 0.15)
    if _longest_run(unvoiced_motion) / cfg.grid_hz >= cfg.silent_motion_s:
        flags.append("mouth_motion_without_speech")
        reasons.append("lips moved for a sustained period with no matching audio")

    score = max(0.0, r)
    if abs(offset_s) > cfg.max_offset_s:
        score *= float(np.exp(-(abs(offset_s) - cfg.max_offset_s) / 0.1))
    if "speech_without_mouth_motion" in flags:
        score = 0.0
    decision: Decision = "match"
    if r < cfg.min_correlation:
        reasons.append(f"mouth movement does not follow the speech (r={r:.2f})")
        decision = "mismatch"
    elif prominence is not None and prominence < cfg.min_prominence:
        reasons.append(f"alignment is no stronger than chance (prominence {prominence:.1f})")
        decision = "mismatch"
        score *= 0.5
    if {"speech_without_mouth_motion", "audio_video_offset"} & set(flags):
        decision = "mismatch"
    return AVSyncResult(
        decision,
        round(score, 4),
        round(r, 4),
        round(offset_s * 1000, 1),
        round(voiced_s, 2),
        round(coverage, 3),
        flags,
        reasons,
        None if prominence is None else round(prominence, 2),
    )
