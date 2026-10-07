"""Synthetic media for API tests only (never used for accuracy claims)."""

from __future__ import annotations

import base64
import io
import wave

import numpy as np
from PIL import Image


def wav_b64(audio: np.ndarray, sr: int = 16000) -> str:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes())
    return base64.b64encode(buf.getvalue()).decode()


def tone(freq: float, seconds: float = 3.0, sr: int = 16000, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * sr)) / sr
    return (0.3 * np.sin(2 * np.pi * freq * t) + 0.01 * rng.normal(0, 1, t.size)).astype(np.float32)


def png_b64(color=(200, 50, 50), size=64, seed=0) -> str:
    rng = np.random.default_rng(seed)
    arr = np.clip(np.full((size, size, 3), color) + rng.integers(0, 4, (size, size, 3)), 0, 255).astype(
        np.uint8
    )
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def syllables(rng, seconds=6.0):
    n = int(seconds * 100)
    env = np.zeros(n)
    t = 30
    while t < n - 30:
        d, g = int(rng.uniform(12, 30)), int(rng.uniform(5, 25))
        env[t : t + d] = rng.uniform(0.6, 1.0)
        t += d + g
    k = np.hanning(9)
    return np.convolve(env, k / k.sum(), mode="same")


def speech_and_mouth(seed=0, seconds=6.0, mismatch=False):
    rng = np.random.default_rng(seed)
    env = syllables(rng, seconds)
    sr = 16000
    n = int(seconds * sr)
    tt = np.arange(n) / sr
    e = np.interp(tt, np.arange(env.size) / 100, env)
    audio = (0.3 * e * (np.sin(2 * np.pi * 140 * tt) + 0.3 * rng.normal(0, 1, n))).astype(np.float32)
    mouth_env = syllables(np.random.default_rng(seed + 100), seconds) if mismatch else env
    times = np.arange(0, seconds, 1 / 30)
    values = 0.03 + 0.35 * np.interp(times, np.arange(mouth_env.size) / 100, mouth_env)
    return audio, times.tolist(), values.tolist()


EYE_L = {
    33: (0.40, 0.40),
    160: (0.42, 0.39),
    158: (0.44, 0.39),
    133: (0.46, 0.40),
    153: (0.44, 0.41),
    144: (0.42, 0.41),
}
EYE_R = {
    362: (0.54, 0.40),
    385: (0.56, 0.39),
    387: (0.58, 0.39),
    263: (0.60, 0.40),
    373: (0.58, 0.41),
    380: (0.56, 0.41),
}
LIPS = {
    13: (0.50, 0.55),
    14: (0.50, 0.56),
    82: (0.48, 0.55),
    87: (0.48, 0.56),
    312: (0.52, 0.55),
    317: (0.52, 0.56),
}


def face_pose(pose="neutral"):
    lm = np.full((478, 2), 0.5)
    for d in (EYE_L, EYE_R, LIPS):
        for k, v in d.items():
            lm[k] = v
    lm[1] = (0.50, 0.48)
    if pose == "blink":
        lm[[160, 158, 153, 144, 385, 387, 373, 380], 1] = 0.40
    elif pose == "turn_left":
        lm[1, 0] += 0.04
    elif pose == "turn_right":
        lm[1, 0] -= 0.04
    elif pose == "open_mouth":
        lm[[14, 87, 317], 1] += 0.06
    return lm.round(4).tolist()


def liveness_series(nonce, steps, fps=15):
    poses = ["neutral"] * fps
    for s in steps:
        poses += [s] * 4 + ["neutral"] * 9
    return {
        "nonce": nonce,
        "frames": [{"t": i / fps, "landmarks": face_pose(p), "faces": 1} for i, p in enumerate(poses)],
    }
