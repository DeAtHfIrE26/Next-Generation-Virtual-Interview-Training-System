"""Build configured backends from environment variables. Returns ``None`` when unset."""

from __future__ import annotations

import os

from interview_core.face.embedding import FaceEmbedder
from interview_core.voice.matching import SpeakerEmbedder


def _float_env(name: str) -> float | None:
    raw = os.getenv(name, "").strip()
    return float(raw) if raw else None


def face_threshold() -> float | None:
    return _float_env("FACE_MATCH_THRESHOLD")


def voice_threshold() -> float | None:
    return _float_env("VOICE_MATCH_THRESHOLD")


def face_embedder() -> FaceEmbedder | None:
    kind = os.getenv("FACE_EMBEDDER", "none").strip().lower()
    if kind == "onnx":
        from interview_core.adapters.onnx_face import OnnxFaceEmbedder

        path = os.getenv("FACE_ONNX_MODEL_PATH", "")
        if not path:
            raise RuntimeError("FACE_EMBEDDER=onnx requires FACE_ONNX_MODEL_PATH")
        return OnnxFaceEmbedder(path)
    if kind in ("", "none"):
        return None
    raise RuntimeError(f"unknown FACE_EMBEDDER={kind!r}")


def speaker_embedder() -> SpeakerEmbedder | None:
    kind = os.getenv("SPEAKER_EMBEDDER", "none").strip().lower()
    if kind == "speechbrain_ecapa":
        from interview_core.adapters.speaker import SpeechBrainEcapa

        return SpeechBrainEcapa()
    if kind == "resemblyzer":
        from interview_core.adapters.speaker import ResemblyzerEncoder

        return ResemblyzerEncoder()
    if kind in ("", "none"):
        return None
    raise RuntimeError(f"unknown SPEAKER_EMBEDDER={kind!r}")
