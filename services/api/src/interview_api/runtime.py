"""Lazily constructed providers (LLM, biometrics, speech, code runner, key provider).

Each returns ``None`` when not configured; the API then degrades honestly (offline question
bank, "verification not configured", browser speech) instead of faking results.
"""

from __future__ import annotations

from functools import cache

from interview_core.adapters import factory
from interview_core.codeexec.runner import Judge0Client
from interview_core.crypto import KeyProvider, key_provider_from_env
from interview_core.nlp import providers
from interview_core.speech import asr, tts


@cache
def llm_provider():
    return providers.from_env()


@cache
def face_embedder():
    return factory.face_embedder()


@cache
def speaker_embedder():
    return factory.speaker_embedder()


@cache
def asr_provider():
    return asr.from_env()


@cache
def tts_provider():
    return tts.from_env()


@cache
def judge0():
    return Judge0Client.from_env()


@cache
def key_provider() -> KeyProvider:
    return key_provider_from_env()


def capabilities() -> dict:
    """What this deployment can actually do, shown to the client and on /health."""
    return {
        "llm": getattr(llm_provider(), "name", None),
        "face_verification": face_embedder() is not None and factory.face_threshold() is not None,
        "voice_verification": speaker_embedder() is not None and factory.voice_threshold() is not None,
        "server_asr": getattr(asr_provider(), "name", None),
        "server_tts": getattr(tts_provider(), "name", None),
        "code_execution": judge0() is not None,
        "neural_avatar": _neural_avatar_enabled(),
    }


def _neural_avatar_enabled() -> bool:
    from interview_api.settings import get_settings

    s = get_settings()
    return bool(s.feature_neural_avatar and s.neural_avatar_url)


def clear() -> None:
    for f in (
        llm_provider,
        face_embedder,
        speaker_embedder,
        asr_provider,
        tts_provider,
        judge0,
        key_provider,
    ):
        f.cache_clear()
