"""Small numeric helpers shared by the audio-visual modules (numpy only)."""

from __future__ import annotations

import numpy as np


def bandpass_fft(x: np.ndarray, sr: int, lo_hz: float, hi_hz: float) -> np.ndarray:
    """Zero-phase band-pass by FFT masking. Fine for utterance-length signals."""
    if x.size == 0:
        return x
    spec = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(x.size, 1 / sr)
    spec[(freqs < lo_hz) | (freqs > hi_hz)] = 0
    return np.fft.irfft(spec, n=x.size).astype(np.float32)


def rms_envelope(x: np.ndarray, sr: int, rate_hz: float = 100.0, win_s: float = 0.025) -> np.ndarray:
    """Frame RMS sampled at ``rate_hz`` (hop = sr/rate_hz, window = win_s)."""
    hop = max(1, round(sr / rate_hz))
    win = max(hop, round(sr * win_s))
    if x.size < win:
        return np.zeros(0, dtype=np.float64)
    n = 1 + (x.size - win) // hop
    idx = np.arange(win)[None, :] + hop * np.arange(n)[:, None]
    frames = x[idx].astype(np.float64)
    return np.sqrt(np.mean(frames**2, axis=1))


def moving_average(x: np.ndarray, k: int) -> np.ndarray:
    if k <= 1 or x.size == 0:
        return x.astype(np.float64)
    kernel = np.ones(k) / k
    return np.convolve(np.pad(x, (k // 2, k - 1 - k // 2), mode="edge"), kernel, mode="valid")


def zscore(x: np.ndarray) -> np.ndarray:
    sd = x.std()
    return (x - x.mean()) / sd if sd > 1e-12 else np.zeros_like(x, dtype=np.float64)


def resample_series(times: np.ndarray, values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Linear interpolation of an irregularly sampled series onto ``grid`` (seconds)."""
    order = np.argsort(times)
    return np.interp(grid, times[order], values[order])


def lagged_correlation(a: np.ndarray, b: np.ndarray, max_lag: int) -> tuple[np.ndarray, np.ndarray]:
    """Pearson correlation of ``a[t]`` with ``b[t + lag]`` for lag in [-max_lag, max_lag].

    Positive lag means ``b`` happens later than ``a``.
    """
    lags = np.arange(-max_lag, max_lag + 1)
    out = np.full(lags.size, np.nan)
    n = min(a.size, b.size)
    for i, lag in enumerate(lags):
        if lag >= 0:
            x, y = a[: n - lag], b[lag:n]
        else:
            x, y = a[-lag:n], b[: n + lag]
        if x.size < 3:
            continue
        sx, sy = x.std(), y.std()
        if sx < 1e-12 or sy < 1e-12:
            out[i] = 0.0
            continue
        out[i] = float(np.mean((x - x.mean()) * (y - y.mean())) / (sx * sy))
    return lags, out
