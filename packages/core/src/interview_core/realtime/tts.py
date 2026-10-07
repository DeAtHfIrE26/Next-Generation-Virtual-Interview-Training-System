"""Streaming text-to-speech for the interviewer's voice.

``synthesize(text)`` yields :class:`TTSChunk` objects (PCM16 mono at ``sample_rate``) sentence by
sentence, so the first sentence starts playing while later ones are still being generated.
Providers that return timing information attach :class:`Mark` objects (word or viseme, seconds
from the start of the utterance) which drive the avatar's lip-sync timeline; providers without
timings (local Kokoro) rely on the browser's audio-driven lip-sync (docs/DECISIONS.md D4).

Providers:
- ``kokoro`` (default, local, free): Kokoro-82M v1.0 via sherpa-onnx.
- ``polly``: Amazon Polly neural/generative, PCM plus viseme and word speech marks.
- ``elevenlabs``: ElevenLabs streaming with character-level timestamps.
"""

from __future__ import annotations

import base64
import json
import os
import re
import threading
from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass, field
from functools import cache

import numpy as np

from interview_core.realtime import assets


@dataclass
class Mark:
    t: float  # seconds from the start of the utterance
    kind: str  # "word" | "viseme"
    value: str
    duration: float | None = None


@dataclass
class TTSChunk:
    pcm16: bytes
    sample_rate: int
    text: str  # the sentence this audio speaks
    marks: list[Mark] = field(default_factory=list)  # relative to the start of this chunk


@dataclass(frozen=True)
class Voice:
    id: str
    label: str
    provider_voice: str  # provider-specific id (Kokoro speaker id, Polly VoiceId, ElevenLabs voice id)


class TTSProvider(ABC):
    name: str
    sample_rate: int
    voices: dict[str, Voice]

    @abstractmethod
    def synthesize(self, text: str, voice: str | None = None) -> Iterator[TTSChunk]: ...

    def voice(self, voice: str | None) -> Voice:
        if voice and voice in self.voices:
            return self.voices[voice]
        return next(iter(self.voices.values()))


_ABBREV = re.compile(r"\b(e\.g|i\.e|etc|vs|Mr|Mrs|Ms|Dr|Prof|Inc|Ltd|approx)\.$", re.IGNORECASE)


def split_sentences(text: str, max_chars: int = 220) -> list[str]:
    """Sentence chunks for low-latency synthesis: split on . ! ? (not after common abbreviations);
    over-long sentences are split at the last comma or space before ``max_chars``."""
    text = " ".join(text.split())
    out: list[str] = []
    buf = ""
    for tok in re.split(r"(?<=[.!?])\s+", text):
        buf = f"{buf} {tok}".strip() if buf else tok
        if _ABBREV.search(buf):
            continue
        out.append(buf)
        buf = ""
    if buf:
        out.append(buf)
    final: list[str] = []
    for s in out:
        while len(s) > max_chars:
            cut = max(s.rfind(", ", 0, max_chars), s.rfind(" ", 0, max_chars))
            cut = cut if cut > 40 else max_chars
            final.append(s[: cut + 1].strip())
            s = s[cut + 1 :].strip()
        if s:
            final.append(s)
    return final


def float_to_pcm16(x: np.ndarray) -> bytes:
    return (np.clip(x, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()


# ----------------------------------------------------------------------------- Kokoro (local)

KOKORO_VOICES = {
    "maya": Voice("maya", "Maya (US, warm)", "3"),  # af_heart
    "daniel": Voice("daniel", "Daniel (UK, calm)", "24"),  # bm_daniel
    "emma": Voice("emma", "Emma (UK, crisp)", "21"),  # bf_emma
    "michael": Voice("michael", "Michael (US, steady)", "16"),  # am_michael
    "priya": Voice("priya", "Priya (Hindi voice speaking English)", "31"),  # hf_alpha
    "arjun": Voice("arjun", "Arjun (Hindi voice speaking English)", "33"),  # hm_omega
    "ananya": Voice("ananya", "Ananya (Hindi voice speaking English)", "32"),  # hf_beta
}


class KokoroTTS(TTSProvider):
    name = "kokoro"
    sample_rate = 24000
    voices = KOKORO_VOICES

    def __init__(self, threads: int | None = None, speed: float | None = None):
        import sherpa_onnx  # optional extra: interview-core[local-speech]

        m = assets.require("tts-kokoro")
        cfg = sherpa_onnx.OfflineTtsConfig(
            model=sherpa_onnx.OfflineTtsModelConfig(
                kokoro=sherpa_onnx.OfflineTtsKokoroModelConfig(
                    model=str(m / "model.onnx"),
                    voices=str(m / "voices.bin"),
                    tokens=str(m / "tokens.txt"),
                    data_dir=str(m / "espeak-ng-data"),
                    lexicon=str(m / "lexicon-us-en.txt"),
                ),
                num_threads=threads or int(os.getenv("TTS_THREADS", "2")),
            ),
            max_num_sentences=1,
        )
        self.tts = sherpa_onnx.OfflineTts(cfg)
        self.sample_rate = self.tts.sample_rate
        self.speed = speed or float(os.getenv("TTS_SPEED", "1.0"))
        self.lock = threading.Lock()

    def synthesize(self, text: str, voice: str | None = None) -> Iterator[TTSChunk]:
        sid = int(self.voice(voice).provider_voice)
        for sentence in split_sentences(text):
            with self.lock:
                audio = self.tts.generate(sentence, sid=sid, speed=self.speed)
            x = np.asarray(audio.samples, dtype=np.float32)
            if x.size:
                yield TTSChunk(float_to_pcm16(x), audio.sample_rate, sentence)


@cache
def kokoro_provider() -> KokoroTTS:
    return KokoroTTS()


# ----------------------------------------------------------------------------- Amazon Polly

POLLY_VOICES = {
    "kajal": Voice("kajal", "Kajal (Indian English)", "Kajal"),
    "ruth": Voice("ruth", "Ruth (US)", "Ruth"),
    "stephen": Voice("stephen", "Stephen (US)", "Stephen"),
    "amy": Voice("amy", "Amy (UK)", "Amy"),
}


class PollyTTS(TTSProvider):
    name = "polly"
    sample_rate = 16000
    voices = POLLY_VOICES

    def __init__(self, engine: str = "neural", region: str | None = None, client=None):
        if client is None:
            import boto3  # optional extra: interview-core[aws]

            client = boto3.client("polly", region_name=region or os.getenv("AWS_REGION", "ap-south-1"))
        self.client, self.engine = client, engine

    def synthesize(self, text: str, voice: str | None = None) -> Iterator[TTSChunk]:
        v = self.voice(voice).provider_voice
        for sentence in split_sentences(text):
            common = {"Text": sentence, "VoiceId": v, "Engine": self.engine}
            pcm = self.client.synthesize_speech(OutputFormat="pcm", SampleRate="16000", **common)[
                "AudioStream"
            ].read()
            raw = self.client.synthesize_speech(
                OutputFormat="json", SpeechMarkTypes=["viseme", "word"], **common
            )
            marks = []
            for line in raw["AudioStream"].read().decode().splitlines():
                if line.strip():
                    m = json.loads(line)
                    marks.append(Mark(m["time"] / 1000.0, m["type"], m["value"]))
            yield TTSChunk(pcm, 16000, sentence, marks)


# ----------------------------------------------------------------------------- ElevenLabs


class ElevenLabsTTS(TTSProvider):
    name = "elevenlabs"
    sample_rate = 24000
    URL = "https://api.elevenlabs.io/v1/text-to-speech/{voice}/stream/with-timestamps"

    def __init__(self, api_key: str, voice_id: str, model: str = "eleven_flash_v2_5"):
        self.key, self.model = api_key, model
        self.voices = {"default": Voice("default", "ElevenLabs voice", voice_id)}

    def synthesize(self, text: str, voice: str | None = None) -> Iterator[TTSChunk]:
        import httpx

        vid = self.voice(voice).provider_voice
        for sentence in split_sentences(text):
            with httpx.stream(
                "POST",
                self.URL.format(voice=vid),
                params={"output_format": "pcm_24000"},
                headers={"xi-api-key": self.key},
                json={"text": sentence, "model_id": self.model},
                timeout=30,
            ) as r:
                r.raise_for_status()
                pcm = bytearray()
                chars: list[str] = []
                starts: list[float] = []
                for line in r.iter_lines():
                    if not line.strip():
                        continue
                    d = json.loads(line)
                    if d.get("audio_base64"):
                        pcm += base64.b64decode(d["audio_base64"])
                    al = d.get("alignment") or {}
                    chars += al.get("characters", [])
                    starts += al.get("character_start_times_seconds", [])
            yield TTSChunk(bytes(pcm), 24000, sentence, _words_from_chars(chars, starts))


def _words_from_chars(chars: list[str], starts: list[float]) -> list[Mark]:
    marks, cur, t0 = [], "", None
    for c, t in zip(chars, starts, strict=False):
        if c.isspace():
            if cur:
                marks.append(Mark(t0 or 0.0, "word", cur))
            cur, t0 = "", None
        else:
            t0 = t if t0 is None else t0
            cur += c
    if cur:
        marks.append(Mark(t0 or 0.0, "word", cur))
    return marks


# ----------------------------------------------------------------------------- selection


def from_env() -> TTSProvider | None:
    kind = os.getenv("TTS_PROVIDER", "kokoro").strip().lower()
    if kind == "none":
        return None
    if kind == "kokoro":
        return kokoro_provider()
    if kind == "polly":
        return PollyTTS(os.getenv("POLLY_ENGINE", "neural"))
    if kind == "elevenlabs":
        key, vid = os.getenv("ELEVENLABS_API_KEY", ""), os.getenv("ELEVENLABS_VOICE_ID", "")
        if not key or not vid:
            raise RuntimeError("TTS_PROVIDER=elevenlabs needs ELEVENLABS_API_KEY and ELEVENLABS_VOICE_ID")
        return ElevenLabsTTS(key, vid, os.getenv("ELEVENLABS_MODEL", "eleven_flash_v2_5"))
    raise RuntimeError(f"unknown TTS_PROVIDER={kind!r} (use kokoro, polly or elevenlabs)")
