"""Small loaders for evaluation media (no heavyweight dependencies)."""

from __future__ import annotations

import json
import wave
from pathlib import Path
from typing import Any

import numpy as np


def read_wav_mono(path: Path) -> tuple[np.ndarray, int]:
    """Return float32 samples in [-1, 1] and the sample rate. 16-bit PCM only."""
    with wave.open(str(path), "rb") as w:
        if w.getsampwidth() != 2:
            raise ValueError(f"{path}: expected 16-bit PCM")
        sr, ch, n = w.getframerate(), w.getnchannels(), w.getnframes()
        pcm = np.frombuffer(w.readframes(n), dtype=np.int16).astype(np.float32) / 32768.0
    if ch > 1:
        pcm = pcm.reshape(-1, ch).mean(axis=1)
    return pcm, sr


def write_wav_mono(path: Path, samples: np.ndarray, sr: int) -> None:
    pcm = (np.clip(samples, -1, 1) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_gray(path: Path) -> np.ndarray:
    import cv2  # opencv-python-headless (Apache-2.0)

    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"cannot read image {path}")
    return img


def read_bgr(path: Path) -> np.ndarray:
    import cv2

    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"cannot read image {path}")
    return img
