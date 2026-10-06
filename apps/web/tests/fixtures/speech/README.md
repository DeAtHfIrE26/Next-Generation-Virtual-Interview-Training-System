# Spoken-answer fixtures (synthetic)

These WAV files are **synthetic speech**, generated with the project's own local TTS (Kokoro v1.0 via
sherpa-onnx, Apache-2.0) so the E2E tests can speak into a fake microphone. They are not
recordings of real people. Real-voice evaluation (including Indian-English speakers) needs recorded
consented audio, which is listed in `NEEDS_FROM_KASHYAP.md`.

| File | Kokoro voice | Accent label | Content |
|---|---|---|---|
| answer-1-us.wav | af_heart (sid 3) | US English | billing batch job, composite index, 6 h to 40 min |
| answer-2-in.wav | hf_alpha (sid 31) | Hindi voice speaking English | API contract disagreement, design review |
| answer-3-uk.wav | bf_emma (sid 21) | UK English | uncertain answer about latency and caching |
| answer-4-in.wav | hf_beta (sid 32) | Hindi voice speaking English | questions for the interviewer |

24 kHz mono PCM16, with 0.25 s of leading and 0.5 s of trailing silence. Regenerate with
`eval/benchmarks/make_speech_fixtures.py` (same text and voices).
