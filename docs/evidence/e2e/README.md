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

## real-llm/ (2026-10-06, commit 45a3f4a, CI E2E run 37507410380, job `full`)

This is the same spoken-interview E2E, run in CI on Chromium with a real open-weights LLM: Qwen2.5-7B-Instruct on Ollama, CPU only. It ran with `E2E_REQUIRE_LLM=1`, under which the run fails if the interviewer ever falls back to a backup question, or if any answer in the final report is left with offline scoring instead of the LLM rubric. Result: **passed** (spoken interview test, 21 min 57 s).

- `e2e-full-video-4x.webm` is a **4× time-lapse** (560 px, 5 fps) of the whole run. The CI job log only keeps its last 5,000 lines, so the real-time video could not be carried out of CI.
- `01-device-check.png`, `03-barge-in.png`, `04-transcript.png`: the device check, the barge-in, and the live transcript.
- `05-report.png`: the report. Both answers were scored by the LLM rubric, every judgement quotes the candidate's own words, and no "offline scoring" badge appears.

Diagnostics from `03-barge-in.png`:
- LLM: `ollama qwen2.5:7b-instruct`. Emergency questions: 0. Barge-ins: 1. Lip-sync visemes detected: 627.
- LLM turn latency: about 148 s for a follow-up question, and 480 s for the first question, which includes planning on a cold model. The turn gap was 156 s.

These are CPU-only CI figures for a 7B model, a deliberately worst-case host. A GPU or a hosted model answers in seconds.

**Defect found in this run and fixed afterwards.** The model spoke a template placeholder, "Hi [Candidate's Name]". The mock-interview transcripts showed a related defect: the model greeted the candidate with the interviewer persona's own name ("Hi Maya"). The interviewer now never addresses the candidate by name, and validation rejects placeholders and the persona's own name (`address_errors`, see DECISIONS D19 follow-up). This video predates that fix.
