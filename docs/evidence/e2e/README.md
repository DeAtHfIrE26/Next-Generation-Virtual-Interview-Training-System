# E2E evidence

## compose-no-llm/ (2026-10-06, re-run on commit a70ff3e)

The spoken-interview E2E (`apps/web/tests/interview.e2e.ts`, Chromium) was run against a stack started with `docker compose up`. The stack was the production web build, the API, Postgres, and speech models downloaded by the `models` init service with checksums verified. Command: `E2E_BASE_URL=http://localhost:3000 E2E_VIDEO=1 npx playwright test --project=chromium`. Result: **3 passed** (2.1 min). The stack was started from empty volumes with `docker compose up -d --build` (3 min 12 s including the model download; [`compose-up.log`](../compose/compose-up.log)).

- **No LLM was configured.** No key or local model is available in the build sandbox, so every question is a flagged backup question, and the run checks that the badge is shown. The real-LLM run is the `full` job in `.github/workflows/e2e.yml`, which uploads its own video artifact.
- Speech in: synthetic WAVs (Kokoro voices, including a Hindi voice speaking English) played into an injected fake microphone. See `apps/web/tests/fixtures/speech/README.md`.
- Real components: AudioWorklet capture → WebSocket → Silero VAD + Nemotron streaming partials + Parakeet final transcript → interviewer agent (emergency path here) → Kokoro TTS → TalkingHead avatar with audio-driven lip-sync (software WebGL, about 3 fps in this sandbox) → report.
- Files:
  - `spoken-interview.webm`: full video.
  - `01-device-check.png`
  - `03-barge-in.png`: the candidate talked over the interviewer; diagnostics show barge-ins = 1.
  - `04-transcript.png`
  - `05-report.png`
- Diagnostics in that run (`?debug=1` panel, visible in `03-barge-in.png`):
  - TTS first audio: p50 2.4 s (n = 2).
  - Turn gap (end of speech → first interviewer audio): 3.9 s (n = 1).
  - Barge-ins 1; lip-sync visemes detected 729; avatar at 3 fps (software WebGL).

  These were measured on a CPU-only sandbox that was also software-rendering the avatar and running the recogniser, so they are not production latency numbers.
