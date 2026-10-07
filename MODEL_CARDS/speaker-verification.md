# Speaker verification (E3)

- **Purpose:** check that the enrolled person is the one answering.
- **Method:** enrolment from at least 3 freshly generated random phrases (at least 8 s total). The phrase text is checked with ASR (or the client transcript, marked as such), which defends against pre-recorded replay. Each answer is scored against the template by cosine similarity with a calibrated threshold. An un-enrolled session never matches (fixes the prototype's first-answer-becomes-reference defect). Code: `interview_core.voice`.
- **Models:** `SPEAKER_EMBEDDER` = `speechbrain_ecapa` (ECAPA-TDNN on VoxCeleb; check the checkpoint licence) or `resemblyzer` (Apache-2.0). None configured by default.
- **Evaluation:** `speaker_verification` suite (EER; spoof acceptance for replay and TTS clones; by accent). **Not measured.**
- **Known risks:** accent and channel mismatch (laptop vs phone mic), illness, background noise. No trained spoof countermeasure is configured; TTS voice cloning remains a gap.
