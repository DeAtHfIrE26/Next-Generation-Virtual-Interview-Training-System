# Speech recognition

- **Options:** `ASR_PROVIDER=deepgram` (nova-3, en-IN, word timings, filler words kept), `faster_whisper` (self-hosted, MIT weights), or `browser` (the client's own recognition; no word timings).
- **Use:** answer transcripts for evaluation, delivery metrics, the voice-enrolment phrase check.
- **Evaluation:** `asr` suite, corpus WER overall and **by accent** (Indian-English regional accents are the primary population). **Not measured.** The prototype's free Google endpoint is not used (not licensed for production).
- **Known risks:** WER differences across accents directly bias content scores; fix before calibrating the evaluator.
