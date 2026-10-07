"""Decoding of client-supplied media. Inputs are size-limited and never written to disk."""

from __future__ import annotations

import base64
import binascii
import io
import wave

import numpy as np
from fastapi import HTTPException
from PIL import Image, UnidentifiedImageError

MAX_IMAGE_BYTES = 400_000
MAX_AUDIO_SECONDS = 120


def _b64(data: str, limit: int) -> bytes:
    if "," in data[:64]:  # tolerate data: URLs
        data = data.split(",", 1)[1]
    try:
        raw = base64.b64decode(data, validate=True)
    except (binascii.Error, ValueError) as e:
        raise HTTPException(422, "invalid base64 media") from e
    if len(raw) > limit:
        raise HTTPException(413, "media too large")
    return raw


def decode_image(data: str) -> np.ndarray:
    raw = _b64(data, MAX_IMAGE_BYTES)
    try:
        img = Image.open(io.BytesIO(raw))
        img.verify()
        img = Image.open(io.BytesIO(raw)).convert("RGB")
    except (UnidentifiedImageError, OSError) as e:
        raise HTTPException(422, "unsupported image") from e
    if max(img.size) > 1024:
        img.thumbnail((1024, 1024))
    return np.asarray(img, dtype=np.uint8)


def decode_wav(data: str) -> tuple[np.ndarray, int]:
    raw = _b64(data, 16000 * 2 * MAX_AUDIO_SECONDS + 1024)
    try:
        with wave.open(io.BytesIO(raw), "rb") as w:
            if w.getsampwidth() != 2 or w.getnchannels() not in (1, 2):
                raise HTTPException(422, "audio must be 16-bit PCM WAV")
            sr, ch = w.getframerate(), w.getnchannels()
            pcm = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32) / 32768.0
    except (wave.Error, EOFError) as e:
        raise HTTPException(422, "invalid WAV audio") from e
    if ch == 2:
        pcm = pcm.reshape(-1, 2).mean(axis=1)
    if len(pcm) / sr > MAX_AUDIO_SECONDS:
        raise HTTPException(413, "audio too long")
    return pcm, sr
