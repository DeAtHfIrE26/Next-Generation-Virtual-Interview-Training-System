"""E2 facial recognition: capture facial samples and verify candidate identity."""

from interview_core.face.embedding import FaceEmbedder, FaceVerifier
from interview_core.face.liveness import Challenge, LivenessResult, issue_challenge, verify_challenge
from interview_core.face.quality import QualityReport, assess_quality

__all__ = [
    "Challenge",
    "FaceEmbedder",
    "FaceVerifier",
    "LivenessResult",
    "QualityReport",
    "assess_quality",
    "issue_challenge",
    "verify_challenge",
]
