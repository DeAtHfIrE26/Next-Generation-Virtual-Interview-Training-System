# Speech recognition

- **Default (local, free, CPU):** sherpa-onnx 1.13.8 (Apache-2.0) running three models:
  - **Live captions:** NVIDIA Nemotron speech streaming 0.6B (560 ms chunks, int8).
  - **Final transcript:** NVIDIA Parakeet TDT 0.6B v2 (int8), run on each Silero-VAD segment. Licence CC-BY-4.0, which requires attribution in the product's notices.
  - **Speech segmentation and server-side barge-in:** Silero VAD v5 (MIT).

  The models are downloaded with pinned SHA-256 checksums (`interview_core.realtime.assets`).
- **Licence check pending:** confirm the Nemotron streaming checkpoint's licence terms on its model page before commercial launch. This is listed in `NEEDS_FROM_KASHYAP.md` for legal review.
- **Premium option:** `STT_PROVIDER=deepgram` (nova-3, `en-IN`, word timings) behind the same interface.
- **Use:** answer transcripts for evaluation, live captions, delivery metrics, end-of-turn detection, and the voice-enrolment phrase check.
- **Measured** (`eval/benchmarks/stt_bench.py`, results in `docs/evidence/voice/stt_bench.txt`):
  - Test set: 30 **synthetic** utterances in 5 Kokoro voices, including two Hindi voices speaking English.
  - Nemotron streaming: WER 2.4%, real-time factor (RTF) 0.33.
  - Parakeet final pass: WER 0.0%, RTF 0.074.

  This is synthetic TTS speech. It is **not** evidence of accuracy on real speakers.
- **Not measured:** WER on real, consented speech by accent. Indian-English regional accents are the primary population. The `asr` evaluation suite needs recorded data (`eval/DATA_COLLECTION.md`).
- **Known risks:**
  - WER differences across accents bias content scores directly. Measure and fix this before calibrating the evaluator.
  - Synthetic-voice results overstate real-world accuracy.
