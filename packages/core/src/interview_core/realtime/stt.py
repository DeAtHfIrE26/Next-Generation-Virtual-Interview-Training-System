"""Streaming speech-to-text for the live interview.

Interface: one :class:`STTSession` per candidate turn. ``accept()`` takes 16 kHz mono PCM16 as it
arrives and returns events: ``partial`` (live caption of the whole turn so far) and ``final``
(a closed speech segment's settled text). ``finalize()`` closes the turn and returns the full
transcript with word timings. ``speaking`` / ``silence_ms`` expose voice activity so the
conversation controller can decide when the candidate has finished.

Providers:
- ``sherpa`` (default, local, free): Nemotron streaming 0.6B for partials; each VAD segment is
  re-decoded with Parakeet TDT 0.6B v2 as soon as it closes (see docs/DECISIONS.md D1).
- ``deepgram`` (premium): Nova-3 live streaming over WebSocket.
"""

from __future__ import annotations

import contextlib
import json
import os
import queue
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import cache
from typing import Literal

import numpy as np

from interview_core.realtime import assets
from interview_core.speech.asr import Word

SAMPLE_RATE = 16000


@dataclass
class STTEvent:
    kind: Literal["partial", "final"]
    text: str  # partial: whole turn so far; final: this segment only
    segment: int = 0
    words: list[Word] = field(default_factory=list)


@dataclass
class TurnTranscript:
    text: str
    words: list[Word]
    audio_seconds: float
    provider: str
    segments: int


class STTSession(ABC):
    provider: str

    @abstractmethod
    def accept(self, pcm16: bytes) -> list[STTEvent]: ...

    @abstractmethod
    def finalize(self) -> tuple[list[STTEvent], TurnTranscript]: ...

    @property
    @abstractmethod
    def speaking(self) -> bool: ...

    @property
    @abstractmethod
    def silence_ms(self) -> float:
        """Milliseconds since the candidate last produced speech (0 while speaking)."""

    @property
    @abstractmethod
    def heard_speech(self) -> bool: ...

    def close(self) -> None:  # noqa: B027 - optional hook
        pass


class STTProvider(ABC):
    name: str

    @abstractmethod
    def open(self, language: str = "en") -> STTSession: ...

    def gate(self) -> SpeechGate | None:
        """A voice-activity gate for barge-in detection, or None if this provider has none."""
        if not assets.is_present("vad"):
            return None
        try:
            return SpeechGate(str(assets.require("vad")))
        except ImportError:  # sherpa-onnx not installed
            return None


class SpeechGate:
    """Server-side barge-in detector (Silero VAD) for the candidate's mic stream while the
    interviewer is speaking. ``accept`` returns True once, when sustained speech starts. (The caller
    keeps the recent audio so the new turn starts with the words that interrupted.)"""

    def __init__(self, vad_model: str, threshold: float = 0.6, min_speech_s: float = 0.3):
        import sherpa_onnx  # optional extra: interview-core[local-speech]

        cfg = sherpa_onnx.VadModelConfig()
        cfg.silero_vad.model = vad_model
        cfg.silero_vad.threshold = threshold
        cfg.silero_vad.min_silence_duration = 0.25
        cfg.silero_vad.min_speech_duration = min_speech_s
        cfg.sample_rate = SAMPLE_RATE
        self.vad = sherpa_onnx.VoiceActivityDetector(cfg, buffer_size_in_seconds=10)
        self._window = np.zeros(0, dtype=np.float32)
        self.fired = False

    def accept(self, pcm16: bytes) -> bool:
        if self.fired:
            return False
        self._window = np.concatenate([self._window, pcm16_to_float(pcm16)])
        n = (self._window.size // 512) * 512
        for i in range(0, n, 512):
            self.vad.accept_waveform(self._window[i : i + 512])
        self._window = self._window[n:]
        while not self.vad.empty():  # segments are not needed, only the onset
            self.vad.pop()
        if self.vad.is_speech_detected():
            self.fired = True
            return True
        return False


def pcm16_to_float(pcm16: bytes) -> np.ndarray:
    return np.frombuffer(pcm16, dtype="<i2").astype(np.float32) / 32768.0


# ----------------------------------------------------------------------------- sherpa-onnx (local)


class SherpaSTT(STTProvider):
    """Local, free, CPU. Models are loaded once per process and shared by all sessions."""

    name = "sherpa"

    def __init__(self, threads: int | None = None, final_pass: bool = True):
        import sherpa_onnx  # optional extra: interview-core[local-speech]

        self._so = sherpa_onnx
        n = threads or int(os.getenv("STT_THREADS", "2"))
        s = assets.require("stt-streaming")
        self.online = sherpa_onnx.OnlineRecognizer.from_transducer(
            tokens=str(s / "tokens.txt"),
            encoder=str(s / "encoder.int8.onnx"),
            decoder=str(s / "decoder.int8.onnx"),
            joiner=str(s / "joiner.int8.onnx"),
            num_threads=n,
            decoding_method="greedy_search",
        )
        self.offline = None
        if final_pass:
            f = assets.require("stt-final")
            self.offline = sherpa_onnx.OfflineRecognizer.from_transducer(
                tokens=str(f / "tokens.txt"),
                encoder=str(f / "encoder.int8.onnx"),
                decoder=str(f / "decoder.int8.onnx"),
                joiner=str(f / "joiner.int8.onnx"),
                num_threads=n,
                decoding_method="greedy_search",
                model_type="nemo_transducer",
            )
        self.vad_model = str(assets.require("vad"))
        self.lock = threading.Lock()  # decoding is serialised per recogniser

    def open(self, language: str = "en") -> STTSession:
        return _SherpaSession(self)


class _SherpaSession(STTSession):
    provider = "sherpa"
    MIN_SILENCE_S = 0.35  # a pause this long closes a segment for the final pass

    def __init__(self, p: SherpaSTT):
        so = p._so
        self.p = p
        self.stream = p.online.create_stream()
        cfg = so.VadModelConfig()
        cfg.silero_vad.model = p.vad_model
        cfg.silero_vad.threshold = 0.5
        cfg.silero_vad.min_silence_duration = self.MIN_SILENCE_S
        cfg.silero_vad.min_speech_duration = 0.2
        cfg.silero_vad.max_speech_duration = 25.0
        cfg.sample_rate = SAMPLE_RATE
        self.vad = so.VoiceActivityDetector(cfg, buffer_size_in_seconds=120)
        self.samples = 0  # total samples received this turn
        self.segment_texts: list[str] = []
        self.words: list[Word] = []
        self.live = ""  # streaming hypothesis since the last closed segment
        self._speaking = False
        self._last_speech_sample = 0
        self._heard = False
        self._window = np.zeros(0, dtype=np.float32)

    # -- helpers
    def _running_text(self) -> str:
        return " ".join(t for t in [*self.segment_texts, self.live] if t).strip()

    def _final_pass(self, seg: np.ndarray, start_sample: int) -> str:
        if self.p.offline is None:
            return self.live.strip()
        with self.p.lock:
            s = self.p.offline.create_stream()
            s.accept_waveform(SAMPLE_RATE, seg)
            self.p.offline.decode_stream(s)
            r = s.result
        text = r.text.strip()
        t0 = start_sample / SAMPLE_RATE
        self.words.extend(_words_from_tokens(r.tokens, r.timestamps, t0, len(seg) / SAMPLE_RATE))
        return text

    def _drain_segments(self, events: list[STTEvent]) -> None:
        while not self.vad.empty():
            seg = self.vad.front
            samples = np.asarray(seg.samples, dtype=np.float32)
            text = self._final_pass(samples, seg.start)
            self.vad.pop()
            if text:
                self.segment_texts.append(text)
                events.append(STTEvent("final", text, len(self.segment_texts) - 1))
            # the streaming hypothesis for this stretch is superseded by the final pass
            with self.p.lock:
                self.p.online.reset(self.stream)
            self.live = ""

    # -- interface
    def accept(self, pcm16: bytes) -> list[STTEvent]:
        x = pcm16_to_float(pcm16)
        if x.size == 0:
            return []
        self.samples += x.size
        events: list[STTEvent] = []
        before = self._running_text()
        with self.p.lock:
            self.stream.accept_waveform(SAMPLE_RATE, x)
            while self.p.online.is_ready(self.stream):
                self.p.online.decode_stream(self.stream)
            self.live = self.p.online.get_result(self.stream).strip()
        # Silero needs 512-sample windows
        self._window = np.concatenate([self._window, x])
        n = (self._window.size // 512) * 512
        for i in range(0, n, 512):  # the VAD must be fed exactly one window at a time
            self.vad.accept_waveform(self._window[i : i + 512])
        self._window = self._window[n:]
        detected = self.vad.is_speech_detected()
        if detected:
            self._speaking, self._heard = True, True
            self._last_speech_sample = self.samples
        elif self._speaking:
            self._speaking = False
        self._drain_segments(events)
        now = self._running_text()
        if now and now != before:
            events.append(STTEvent("partial", now, len(self.segment_texts)))
        return events

    def finalize(self) -> tuple[list[STTEvent], TurnTranscript]:
        events: list[STTEvent] = []
        if self._window.size:
            self.vad.accept_waveform(np.pad(self._window, (0, 512 - self._window.size)))
            self._window = np.zeros(0, dtype=np.float32)
        self.vad.flush()
        self._drain_segments(events)
        if self.live:  # speech the VAD never closed (e.g. very short): keep the streaming text
            self.segment_texts.append(self.live)
            self.live = ""
        text = " ".join(self.segment_texts).strip()
        return events, TurnTranscript(
            text, list(self.words), self.samples / SAMPLE_RATE, self.provider, len(self.segment_texts)
        )

    @property
    def speaking(self) -> bool:
        return self._speaking

    @property
    def silence_ms(self) -> float:
        if self._speaking or not self._heard:
            return 0.0
        return (self.samples - self._last_speech_sample) / SAMPLE_RATE * 1000.0

    @property
    def heard_speech(self) -> bool:
        return self._heard


def _words_from_tokens(tokens: list[str], times: list[float], offset: float, seg_dur: float) -> list[Word]:
    """Join BPE tokens ("▁hello", "wor", "ld") into words with start/end times."""
    words: list[Word] = []
    cur, start = "", None
    for i, (tok, t) in enumerate(zip(tokens, times, strict=False)):
        boundary = tok.startswith(("▁", " "))
        piece = tok.replace("▁", "").strip()
        if boundary and cur:
            end = offset + t
            words.append(Word(cur, start or offset, end))
            cur, start = "", None
        if piece:
            if start is None:
                start = offset + t
            cur += piece
        if i == len(tokens) - 1 and cur:
            words.append(Word(cur, start or offset, offset + min(seg_dur, t + 0.3)))
    return [w for w in words if any(c.isalnum() for c in w.word)]


@cache
def sherpa_provider() -> SherpaSTT:
    return SherpaSTT()


# ----------------------------------------------------------------------------- Deepgram (premium)


class DeepgramSTT(STTProvider):
    """Deepgram live transcription (Nova-3). Needs DEEPGRAM_API_KEY."""

    name = "deepgram"
    URL = "wss://api.deepgram.com/v1/listen"

    def __init__(self, api_key: str, model: str = "nova-3", language: str = "en-IN"):
        self.key, self.model, self.language = api_key, model, language

    def open(self, language: str = "en") -> STTSession:
        lang = self.language if language == "en" else language
        return _DeepgramSession(self, lang)


class _DeepgramSession(STTSession):
    provider = "deepgram"

    def __init__(self, p: DeepgramSTT, language: str):
        from websockets.sync.client import connect  # optional dependency

        params = (
            f"model={p.model}&language={language}&encoding=linear16&sample_rate={SAMPLE_RATE}&channels=1"
            "&interim_results=true&punctuate=true&smart_format=true&vad_events=true&endpointing=300"
        )
        self.ws = connect(f"{p.URL}?{params}", additional_headers={"Authorization": f"Token {p.key}"})
        self.q: queue.Queue[dict] = queue.Queue()
        self.finals: list[str] = []
        self.words: list[Word] = []
        self.interim = ""
        self.samples = 0
        self._last_speech = None
        self._heard = False
        self._closed = False
        self.reader = threading.Thread(target=self._read, daemon=True)
        self.reader.start()

    def _read(self) -> None:
        try:
            for raw in self.ws:
                if isinstance(raw, str):
                    self.q.put(json.loads(raw))
        except Exception:
            self.q.put({"type": "_closed"})

    def _drain(self) -> list[STTEvent]:
        events: list[STTEvent] = []
        while True:
            try:
                m = self.q.get_nowait()
            except queue.Empty:
                break
            if m.get("type") == "_closed":
                self._closed = True
                continue
            if m.get("type") == "SpeechStarted":
                self._heard, self._last_speech = True, time.monotonic()
                continue
            if m.get("type") != "Results":
                continue
            alt = (m.get("channel", {}).get("alternatives") or [{}])[0]
            text = (alt.get("transcript") or "").strip()
            if text:
                self._heard, self._last_speech = True, time.monotonic()
            if m.get("is_final"):
                if text:
                    self.finals.append(text)
                    self.words.extend(
                        Word(w.get("punctuated_word", w["word"]), w["start"], w["end"], w.get("confidence"))
                        for w in alt.get("words", [])
                    )
                    events.append(STTEvent("final", text, len(self.finals) - 1))
                self.interim = ""
            else:
                self.interim = text
            running = " ".join([*self.finals, self.interim]).strip()
            if running:
                events.append(STTEvent("partial", running, len(self.finals)))
        return events

    def accept(self, pcm16: bytes) -> list[STTEvent]:
        if self._closed:
            raise ConnectionError("deepgram connection closed")
        self.samples += len(pcm16) // 2
        self.ws.send(pcm16)
        return self._drain()

    def finalize(self) -> tuple[list[STTEvent], TurnTranscript]:
        with contextlib.suppress(Exception):
            self.ws.send(json.dumps({"type": "Finalize"}))
            time.sleep(0.4)  # Deepgram flushes the last interim as a final within a few hundred ms
        events = self._drain()
        return events, TurnTranscript(
            " ".join(self.finals).strip(),
            list(self.words),
            self.samples / SAMPLE_RATE,
            self.provider,
            len(self.finals),
        )

    @property
    def speaking(self) -> bool:
        return self._last_speech is not None and time.monotonic() - self._last_speech < 0.3

    @property
    def silence_ms(self) -> float:
        if self._last_speech is None:
            return 0.0
        return max(0.0, (time.monotonic() - self._last_speech) * 1000.0)

    @property
    def heard_speech(self) -> bool:
        return self._heard

    def close(self) -> None:
        with contextlib.suppress(Exception):
            self.ws.send(json.dumps({"type": "CloseStream"}))
            self.ws.close()


# ----------------------------------------------------------------------------- selection


def from_env() -> STTProvider | None:
    kind = os.getenv("STT_PROVIDER", "sherpa").strip().lower()
    if kind == "none":
        return None
    if kind == "deepgram":
        key = os.getenv("DEEPGRAM_API_KEY", "")
        if not key:
            raise RuntimeError("STT_PROVIDER=deepgram needs DEEPGRAM_API_KEY")
        return DeepgramSTT(
            key, os.getenv("DEEPGRAM_MODEL", "nova-3"), os.getenv("DEEPGRAM_LANGUAGE", "en-IN")
        )
    if kind == "sherpa":
        return sherpa_provider()
    raise RuntimeError(f"unknown STT_PROVIDER={kind!r} (use sherpa or deepgram)")


def transcribe(provider: STTProvider, audio: np.ndarray, sr: int, language: str = "en") -> TurnTranscript:
    """One-shot transcription of a recording through a streaming provider (100 ms chunks)."""
    x = np.asarray(audio, dtype=np.float32)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if sr != SAMPLE_RATE:
        n = int(len(x) * SAMPLE_RATE / sr)
        x = np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x).astype(np.float32)
    pcm = (np.clip(x, -1, 1) * 32767).astype("<i2").tobytes()
    s = provider.open(language)
    try:
        step = SAMPLE_RATE // 10 * 2
        for i in range(0, len(pcm), step):
            s.accept(pcm[i : i + step])
        _, tr = s.finalize()
        return tr
    finally:
        s.close()
