# Third-party components: licences and obligations

This is an engineering inventory. It is **not legal advice**, and every item marked "review" needs a lawyer's confirmation before a public launch (`NEEDS_FROM_KASHYAP.md` §8–9).

## Runtime models and assets

| Component | Used for | Licence | Obligation or status |
|---|---|---|---|
| sherpa-onnx 1.13.8 (k2-fsa) | Local STT, VAD and TTS runtime | Apache-2.0 | Keep the notice |
| NVIDIA Parakeet TDT 0.6B v2 (int8, sherpa-onnx export) | Final transcript per speech segment | CC-BY-4.0 | **Attribution** in product notices |
| NVIDIA Nemotron speech streaming 0.6B (int8, sherpa-onnx export) | Live captions | NVIDIA model licence | **Review**: confirm the terms on the model page |
| Silero VAD v5 | Speech segmentation, server-side barge-in (and the browser VAD) | MIT | Keep the notice |
| Kokoro v1.0 (multi-lang, sherpa-onnx export) | Local interviewer voice | Apache-2.0 | Keep the notice |
| espeak-ng data (bundled with the Kokoro export) | Phonemisation for Kokoro | GPL-3.0 | **Review** before distributing images that contain it; can be avoided with `TTS_PROVIDER=polly` or `elevenlabs` |
| Qwen2.5-7B-Instruct (via Ollama, optional) | Free local interviewer LLM | Apache-2.0 | Keep the notice |
| TalkingHead 1.7.0 | 3D avatar runtime | MIT | Keep the notice |
| HeadAudio 0.1.0 (including its 14 KB model) | Audio-driven visemes | MIT | Keep the notice |
| three.js 0.180 | WebGL | MIT | Keep the notice |
| MPFB-generated character (`apps/web/public/avatars/interviewer.glb`) | Interviewer avatar | CC0 | None (provenance in `apps/web/public/avatars/README.md`) |
| @ricky0123/vad-web 0.0.31, onnxruntime-web | Browser VAD | MIT / MIT | Keep the notices |
| MediaPipe Tasks (Face Landmarker, EfficientDet-Lite0) | On-device E2/E4/E5/E7 signals | Apache-2.0 | Keep the notice |
| Geist fonts | UI typography | SIL OFL 1.1 | Keep the notice |

## Hosted providers (optional, behind the same interfaces)

These are Anthropic, Google Gemini, Deepgram, Amazon Polly and ElevenLabs. Each is used under that provider's commercial terms with the owner's own account. Voice and transcript data go to a provider only when its key is configured, and the privacy notice must list the providers enabled in production.

## Synthetic test data

The E2E speech fixtures and the STT benchmark set are synthetic. They were generated with Kokoro and are labelled as such (`apps/web/tests/fixtures/speech/README.md`, `docs/evidence/voice/`). No recordings of real people are in the repository.
