"""ASR providers returning word timestamps (needed for delivery metrics and E4 alignment).

``browser`` mode means the client's own speech recognition produced the text (no word
timings; delivery metrics that need them are reported as not measured).
"""

from __future__ import annotations

import io
import os
import wave
from dataclasses import dataclass, field
from typing import Protocol

import httpx
import numpy as np


@dataclass(frozen=True)
class Word:
    word: str
    start: float
    end: float
    confidence: float | None = None


@dataclass
class Transcript:
    text: str
    words: list[Word] = field(default_factory=list)
    provider: str = ""
    audio_seconds: float = 0.0


class ASRProvider(Protocol):
    name: str

    def transcribe(self, audio: np.ndarray, sr: int) -> Transcript: ...


def to_wav_bytes(audio: np.ndarray, sr: int) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes())
    return buf.getvalue()


class DeepgramASR:
    """Deepgram pre-recorded transcription (per utterance) with word timings."""

    name = "deepgram"
    URL = "https://api.deepgram.com/v1/listen"

    def __init__(self, api_key: str, model: str = "nova-3", language: str = "en-IN", timeout_s: float = 20.0):
        self._key, self.model, self.language, self.timeout_s = api_key, model, language, timeout_s

    def transcribe(self, audio: np.ndarray, sr: int) -> Transcript:
        r = httpx.post(
            self.URL,
            params={
                "model": self.model,
                "language": self.language,
                "punctuate": "true",
                "smart_format": "true",
                "filler_words": "true",
            },
            headers={"Authorization": f"Token {self._key}", "Content-Type": "audio/wav"},
            content=to_wav_bytes(audio, sr),
            timeout=self.timeout_s,
        )
        r.raise_for_status()
        alt = r.json()["results"]["channels"][0]["alternatives"][0]
        words = [
            Word(w.get("punctuated_word", w["word"]), float(w["start"]), float(w["end"]), w.get("confidence"))
            for w in alt.get("words", [])
        ]
        return Transcript(alt.get("transcript", ""), words, self.name, len(audio) / sr)


class FasterWhisperASR:
    """Self-hosted Whisper (MIT weights) via faster-whisper; CPU or GPU."""

    name = "faster_whisper"

    def __init__(self, model_size: str = "small", device: str = "auto"):
        from faster_whisper import WhisperModel  # optional extra

        self._model = WhisperModel(model_size, device=device)

    def transcribe(self, audio: np.ndarray, sr: int) -> Transcript:
        if sr != 16000:
            n = round(len(audio) * 16000 / sr)
            audio = np.interp(np.linspace(0, len(audio) - 1, n), np.arange(len(audio)), audio).astype(
                np.float32
            )
        segments, _ = self._model.transcribe(audio.astype(np.float32), language="en", word_timestamps=True)
        words, texts = [], []
        for seg in segments:
            texts.append(seg.text.strip())
            words += [
                Word(w.word.strip(), float(w.start), float(w.end), float(w.probability))
                for w in seg.words or []
            ]
        return Transcript(" ".join(texts), words, self.name, len(audio) / 16000)


def from_env() -> ASRProvider | None:
    kind = os.getenv("ASR_PROVIDER", "browser").lower()
    if kind == "deepgram":
        key = os.getenv("DEEPGRAM_API_KEY", "")
        if not key:
            raise RuntimeError("ASR_PROVIDER=deepgram requires DEEPGRAM_API_KEY")
        return DeepgramASR(key, os.getenv("DEEPGRAM_MODEL", "nova-3"))
    if kind == "faster_whisper":
        return FasterWhisperASR(os.getenv("WHISPER_MODEL", "small"))
    return None
