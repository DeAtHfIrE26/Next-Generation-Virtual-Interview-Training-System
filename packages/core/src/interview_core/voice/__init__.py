"""E3 voice authentication: record voice references and perform real-time voice matching."""

from interview_core.voice.matching import (
    SpeakerEmbedder,
    UtteranceCheck,
    VoiceSession,
    VoiceStatus,
    VoiceVerifier,
)
from interview_core.voice.prompts import PhraseChallenge, check_phrase, issue_phrase

__all__ = [
    "PhraseChallenge",
    "SpeakerEmbedder",
    "UtteranceCheck",
    "VoiceSession",
    "VoiceStatus",
    "VoiceVerifier",
    "check_phrase",
    "issue_phrase",
]
