# E2E evidence

## compose-no-llm/ (2026-10-06)

The spoken-interview E2E (`apps/web/tests/interview.e2e.ts`, Chromium) was run against a stack started with `docker compose up`. The stack was the production web build, the API, Postgres, and speech models downloaded by the `models` init service with checksums verified. Command: `E2E_BASE_URL=http://localhost:3000 E2E_VIDEO=1 npx playwright test --project=chromium`. Result: **3 passed** (2.5 min).

- **No LLM was configured.** No key or local model is available in the build sandbox, so every question is a flagged backup question, and the run checks that the badge is shown. The real-LLM run is the `full` job in `.github/workflows/e2e.yml`, which uploads its own video artifact.
- Speech in: synthetic WAVs (Kokoro voices, including a Hindi voice speaking English) played into an injected fake microphone. See `apps/web/tests/fixtures/speech/README.md`.
- Real components: AudioWorklet capture → WebSocket → Silero VAD + Nemotron streaming partials + Parakeet final transcript → interviewer agent (emergency path here) → Kokoro TTS → TalkingHead avatar with audio-driven lip-sync (software WebGL, about 2 fps in this sandbox) → report.
- Files:
  - `spoken-interview.webm`: full video.
  - `01-device-check.png`
  - `03-barge-in.png`: the candidate talked over the interviewer; diagnostics show barge-ins = 1.
  - `04-transcript.png`
  - `05-report.png`
- Diagnostics in that run:
  - TTS first audio: p50 2.4 s.
  - Turn gap (end of speech → first interviewer audio): 3.9 s.

  Both were measured on a CPU-only sandbox that was also software-rendering the avatar and running the recogniser, so they are not production latency numbers.
