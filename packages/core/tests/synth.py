"""Synthetic audio/landmark generators for unit tests only (never for accuracy claims)."""

from __future__ import annotations

import numpy as np


def syllable_envelope(rng: np.random.Generator, seconds: float, rate_hz: float = 100.0) -> np.ndarray:
    n = int(seconds * rate_hz)
    env = np.zeros(n)
    t = int(0.3 * rate_hz)
    while t < n - int(0.3 * rate_hz):
        dur = int(rng.uniform(0.12, 0.3) * rate_hz)
        gap = int(rng.uniform(0.05, 0.25) * rate_hz)
        env[t : t + dur] = rng.uniform(0.6, 1.0)
        t += dur + gap
    k = np.hanning(9)
    return np.convolve(env, k / k.sum(), mode="same")


def speech_audio(
    env_100hz: np.ndarray, sr: int, rng: np.random.Generator, noise: float = 0.002
) -> np.ndarray:
    n = int(len(env_100hz) * sr / 100)
    t = np.arange(n) / sr
    env = np.interp(t, np.arange(len(env_100hz)) / 100, env_100hz)
    voiced = np.sin(2 * np.pi * 140 * t) + 0.5 * np.sin(2 * np.pi * 900 * t) + 0.3 * rng.normal(0, 1, n)
    return (0.3 * env * voiced + noise * rng.normal(0, 1, n)).astype(np.float32)


def mouth_series(
    env_100hz: np.ndarray,
    fps: float,
    lag_s: float,
    rng: np.random.Generator,
    jitter_s: float = 0.0,
    amp: float = 0.35,
) -> tuple[np.ndarray, np.ndarray]:
    """Aperture (inter-ocular units) that follows ``env`` delayed by ``lag_s``."""
    dur = len(env_100hz) / 100
    times = np.arange(0, dur, 1 / fps)
    if jitter_s:
        times = np.sort(times + rng.uniform(-jitter_s, jitter_s, times.size))
        times = times[(times >= 0) & (times < dur)]
    src = np.clip(times - lag_s, 0, dur - 0.01)
    ap = 0.03 + amp * np.interp(src, np.arange(len(env_100hz)) / 100, env_100hz)
    return times, ap + rng.normal(0, 0.01, times.size)
