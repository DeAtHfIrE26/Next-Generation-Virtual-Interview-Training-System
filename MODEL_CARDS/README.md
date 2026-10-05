# Model cards

One card per component that makes a judgement about a person or their answer. Every card shares two facts as of 2026-10-05:

- **Evaluation status: not measured.** No consented evaluation set exists yet. `docs/EVAL_REPORT.md` lists the data each suite needs (`eval/DATA_COLLECTION.md`). No accuracy figure may be quoted until it appears there.
- **Subgroup performance: unknown.** Every suite reports per-subgroup metrics once data with self-reported attributes exists. Gaps above tolerance block release of the affected component.

| Card | Patent element | Licence gate |
|---|---|---|
| [face-verification.md](face-verification.md) | E2 | Needs a commercially licensed embedding model |
| [liveness.md](liveness.md) | E2 (hardening) | None (geometric, no learned model) |
| [speaker-verification.md](speaker-verification.md) | E3 | Check the chosen checkpoint's licence |
| [lipsync-avsync.md](lipsync-avsync.md) | E4 | None (signal processing) |
| [gaze.md](gaze.md) | E5 | MediaPipe Face Landmarker, Apache-2.0 |
| [device-detection.md](device-detection.md) | E7 | MediaPipe EfficientDet-Lite0, Apache-2.0 |
| [interviewer-llm.md](interviewer-llm.md) | E6 | Provider terms |
| [answer-evaluator.md](answer-evaluator.md) | E6 / E9 | Provider terms |
| [asr.md](asr.md) | Speech | Provider terms / Whisper MIT |
| [avatar-tts.md](avatar-tts.md) | Not a patent element | Original artwork; TTS provider terms |
| [legacy-prototype.md](legacy-prototype.md) | All (reference) | Contains AGPL and non-commercial weights: never ship |
