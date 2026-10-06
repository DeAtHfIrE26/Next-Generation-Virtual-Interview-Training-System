"""Lazily constructed providers (LLM chain, streaming speech, biometrics, code runner, key provider).

Speech defaults to the local, free stack (sherpa-onnx streaming STT + Kokoro TTS); premium
providers are selected by configuration. A missing LLM key is not hidden: the interviewer then
uses flagged emergency questions and ``capabilities()`` reports ``llm: null``.
"""

from __future__ import annotations

import logging
import os
import threading
from functools import cache

from interview_core.adapters import factory
from interview_core.agent import llm as chat
from interview_core.codeexec.runner import Judge0Client
from interview_core.crypto import KeyProvider, key_provider_from_env
from interview_core.realtime import stt, tts

log = logging.getLogger("interview_api.runtime")


@cache
def llm_chain() -> tuple:
    try:
        return tuple(chat.chain_from_env())
    except Exception as e:  # misconfiguration is reported, not fatal
        log.error("LLM provider misconfigured: %s", e)
        return ()


@cache
def stt_provider() -> stt.STTProvider | None:
    try:
        return stt.from_env()
    except Exception as e:  # models not downloaded, bad key, ...
        log.error("speech-to-text unavailable: %s", e)
        return None


@cache
def tts_provider() -> tts.TTSProvider | None:
    try:
        return tts.from_env()
    except Exception as e:
        log.error("text-to-speech unavailable: %s", e)
        return None


@cache
def face_embedder():
    return factory.face_embedder()


@cache
def speaker_embedder():
    return factory.speaker_embedder()


@cache
def judge0():
    return Judge0Client.from_env()


@cache
def key_provider() -> KeyProvider:
    return key_provider_from_env()


def capabilities() -> dict:
    """What this deployment can actually do, shown to the client and on /ready."""
    chain = llm_chain()
    s, t = stt_provider(), tts_provider()
    return {
        "llm": chain[0].name if chain else None,
        "llm_model": chain[0].model if chain else None,
        "llm_fallback": chain[1].name if len(chain) > 1 else None,
        "stt": getattr(s, "name", None),
        "tts": getattr(t, "name", None),
        "voices": sorted(t.voices) if t else [],
        "face_verification": face_embedder() is not None and factory.face_threshold() is not None,
        "voice_verification": speaker_embedder() is not None and factory.voice_threshold() is not None,
        "code_execution": judge0() is not None,
        "neural_avatar": _neural_avatar_enabled(),
    }


def warm_up_async() -> None:
    """Load speech models in the background so the first interview turn does not pay for it."""
    if os.getenv("SPEECH_WARMUP", "1") != "1":
        return

    def run() -> None:
        stt_provider()
        tts_provider()

    threading.Thread(target=run, name="speech-warmup", daemon=True).start()


def _neural_avatar_enabled() -> bool:
    from interview_api.settings import get_settings

    s = get_settings()
    return bool(s.feature_neural_avatar and s.neural_avatar_url)


def clear() -> None:
    for f in (llm_chain, stt_provider, tts_provider, face_embedder, speaker_embedder, judge0, key_provider):
        f.cache_clear()
