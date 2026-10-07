"""E4 lip-sync verification: analyse mouth movements and detect speech-authenticity mismatches."""

from interview_core.lipsync.avsync import AVSyncConfig, AVSyncResult, verify_av_sync
from interview_core.lipsync.mouth import mouth_aperture

__all__ = ["AVSyncConfig", "AVSyncResult", "mouth_aperture", "verify_av_sync"]
