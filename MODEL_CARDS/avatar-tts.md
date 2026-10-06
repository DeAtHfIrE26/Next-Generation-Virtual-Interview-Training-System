# Interviewer avatar and voice (not a patent element)

- **Avatar:**
  - TalkingHead 1.7.0 (MIT) on three.js 0.180 (MIT), served from `/vendor`.
  - Character: an MPFB-generated model (CC0; ARKit 52 blendshapes plus Oculus visemes, 67-joint rig), optimised to 4.0 MB (`apps/web/scripts/optimize-avatar.sh`, provenance in `apps/web/public/avatars/README.md`).
  - Lip-sync: HeadAudio (MIT) derives viseme weights from the TTS audio actually playing, in an AudioWorklet. It works with any TTS provider.
  - Behaviour states: blinking, eye contact, idle, listening, thinking and speaking.
  - Quality tiers: high, medium, low and software. With no WebGL, an audio-reactive orb is shown instead.
- **Voice:**
  - Default: Kokoro v1.0 (Apache-2.0) via sherpa-onnx, 24 kHz, sentence-streamed.
  - Personas: Maya (US), Emma (UK), Priya and Ananya (Hindi voices speaking English).
  - Options: Amazon Polly (neural, with viseme marks) and ElevenLabs.
  - **Licence note:** Kokoro's phonemiser data comes from espeak-ng (GPL-3.0). Distributing images that bundle espeak-ng data needs legal review. See `docs/legal/COMPLIANCE_NOTES.md`.
- **Measured:**
  - Kokoro synthesis: RTF median 0.278, max 0.545 (30 utterances, sandbox CPU).
  - First interviewer audio after the question text: p50 1.2–2.4 s on CPU-only machines that were also software-rendering the avatar. These are not production numbers.
  - Avatar: about 1–2 fps under software WebGL in headless CI. Not measured on real GPUs.
- **Not measured:** lip-sync quality (LSE-D/LSE-C), or frame rate and first-frame latency on real devices.
- **Open item:** only one licensed character exists, so all personas are female-presenting (DECISIONS D6). A second MPFB model is needed for male personas.
