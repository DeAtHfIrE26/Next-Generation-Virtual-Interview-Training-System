"""TTS providers that also return viseme timings for the avatar."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Protocol


@dataclass(frozen=True)
class VisemeMark:
    t_ms: int
    viseme: str  # provider viseme symbol


@dataclass
class Speech:
    audio: bytes
    mime: str
    visemes: list[VisemeMark] = field(default_factory=list)
    words: list[tuple[int, str]] = field(default_factory=list)  # (t_ms, word)
    provider: str = ""
    characters: int = 0


class TTSProvider(Protocol):
    name: str

    def synthesize(self, text: str) -> Speech: ...


class PollyTTS:
    """Amazon Polly neural voice: audio plus viseme and word speech marks."""

    name = "polly"

    def __init__(self, voice: str = "Kajal", region: str | None = None, engine: str = "neural"):
        import boto3  # optional extra: interview-core[aws]

        self._client = boto3.client("polly", region_name=region or os.getenv("AWS_REGION", "ap-south-1"))
        self.voice, self.engine = voice, engine

    def synthesize(self, text: str) -> Speech:
        common = {"Text": text, "VoiceId": self.voice, "Engine": self.engine}
        audio = self._client.synthesize_speech(OutputFormat="mp3", **common)["AudioStream"].read()
        marks_raw = (
            self._client.synthesize_speech(OutputFormat="json", SpeechMarkTypes=["viseme", "word"], **common)[
                "AudioStream"
            ]
            .read()
            .decode()
        )
        visemes, words = [], []
        for line in marks_raw.splitlines():
            if not line.strip():
                continue
            m = json.loads(line)
            if m["type"] == "viseme":
                visemes.append(VisemeMark(int(m["time"]), m["value"]))
            elif m["type"] == "word":
                words.append((int(m["time"]), m["value"]))
        return Speech(audio, "audio/mpeg", visemes, words, self.name, len(text))


def from_env() -> TTSProvider | None:
    if os.getenv("TTS_PROVIDER", "browser").lower() == "polly":
        return PollyTTS(os.getenv("POLLY_VOICE", "Kajal"))
    return None
