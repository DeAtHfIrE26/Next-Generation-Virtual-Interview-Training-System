# Truth report

Every line below comes from running the app the way a new user would, or from a command whose output is saved under `docs/evidence/truth/`.

Status values:
- **WORKS:** does what it claims, observed.
- **BROKEN:** fails or produces nothing.
- **FAKE:** looks like it works, but the result is mocked, hardcoded, static, random or not derived from the input.
- **MISSING:** not built.

## Run 1: before the rebuild (2026-10-06, commit `49d0278`)

**How it was run**
- Config: copied `.env.example` to `.env` (offline defaults, no vendor keys), exactly as the README says.
- Browser: Chromium with fake camera and microphone. The microphone was fed a 16.7 s spoken answer: `/tmp/claude-0/audio/answer1.wav`, synthesized locally with a Piper voice.
- Script: `/tmp/claude-0/truth-run.mjs`.
- Output: [`docs/evidence/truth/truth-run.log`](evidence/truth/truth-run.log).
- Screenshots: `docs/evidence/truth/01-landing.png` to `07-room-feedback.png`.

### Starting the app

| Feature | Status | Evidence |
|---|---|---|
| One command: `docker compose up --build` | **BROKEN** here | [`compose-up.log`](evidence/truth/compose-up.log). First attempt: Docker Hub `429 Too Many Requests`. After switching to the `mirror.gcr.io` mirror, the API image still fails, because `services/api/Dockerfile` copies `uv` from `ghcr.io/astral-sh/uv`, which this network blocks. The dependency on ghcr.io is a portability weakness in our Dockerfile. |
| Native start (`uvicorn` + `next start`) | **WORKS** | `curl localhost:8000/ready` returned `{"ok":true,"capabilities":{"llm":null,...,"server_asr":null,"server_tts":null}}`. |

### Core interview loop (what the owner reported)

| Question asked | Answer | Evidence |
|---|---|---|
| Does question text come from a static list? | **YES. FAKE.** | `/next` returned `"source":"bank"`. The question "Describe a disagreement with a colleague. How did you resolve it?" is a verbatim entry of `packages/core/src/interview_core/nlp/question_bank.json` (`in_static_bank: true` in the log). The LLM path exists in code, but the default configuration never uses it. |
| Does the mic actually capture audio? | **YES. WORKS.** | The answer request carried `audio_wav_bytes: 816738` and 145 mouth-aperture samples. |
| Does voice activity detection end the turn on its own? | **UNVERIFIED** | Chromium's fake capture loops the WAV file endlessly, so the silence that would end a turn never occurs. The room's VAD is a simple energy threshold (`lib/vad.ts`). |
| Does speech-to-text return real transcripts? | **NO. BROKEN.** | `"transcript": ""` after 16.7 s of clear speech. No server STT is configured by default (`server_asr: null`). The browser fallback (Web Speech API) produced nothing: in Chrome it depends on Google's cloud service, and Firefox and Safari don't have it at all. No live captions appeared (`caption/transcript text visible: (none)`). |
| Does text-to-speech play? | **NO in this browser. Low quality where it does.** | `speechSynthesis voices: 0`, so the interviewer was silent. The default is the browser's built-in voice, which on desktops is the operating system's robotic voice. Server TTS (Polly) exists, but only with AWS keys. |
| Does the avatar's mouth move in sync with the audio? | **NO. FAKE.** | `distinct avatar mouth states over 3 s: 1`. The avatar is a flat 2D SVG cartoon (`components/Avatar.tsx`, see `05-room-asking.png`). Its mouth follows visemes estimated from word-boundary events; it never looks at the audio. Blinking uses `Math.random`. |
| Is the feedback derived from the answer? | **FAKE in default config** | A score of 25% labelled "experimental" was produced from an **empty** transcript, using structure heuristics only (`07-room-feedback.png`: "content not assessed in this mode"). |

### Everything else

| Feature | Status | Evidence |
|---|---|---|
| Landing, sign-up, consent, dashboard | **WORKS** (visually basic) | `01`–`04` screenshots |
| Interview set-up (role, JD, resume) | **WORKS**, but a thin parameter set | `04-new-interview.png`. There is no company, interview type, round, language or skills-to-probe input. |
| Adaptive difficulty and follow-ups | **FAKE in default config** | Bank questions are picked by difficulty. Follow-ups are generic, because there is no transcript to drill into. |
| Interview blueprint or plan | **MISSING** | — |
| Streaming STT with partial captions | **MISSING** | The answer is uploaded as one WAV after the turn. |
| Streaming TTS | **MISSING** | — |
| Barge-in | **Partial** | The energy VAD cancels browser TTS. Untested with real audio. |
| Mic permission errors, device picker, reconnect | **MISSING** | A permission failure falls back to audio-only, with no device picker and no reconnect. |
| Diagnostics panel (`?debug=1`) | **MISSING** | — |
| 3D avatar, personas, idle/listening/thinking states | **MISSING** | — |
| Candidate-side lip-sync verification (patent E4) | **WORKS** (server; not visible to the user) | Mouth series and audio are sent, and `verify_av_sync` is unit-tested (`packages/core/tests/test_avsync.py`). |
| Face and voice verification (E2, E3) | **MISSING in default config** | They need a licensed face model and a voice model. `/ready` shows `face_verification: false, voice_verification: false`. |
| Gaze and integrity notices (E5, E7) | **WORKS** | "We can't see your face" is correct: the fake camera shows a test pattern, not a face (`07-room-feedback.png`). |
| Coding challenge (E8) | **Partial** | SQL challenges are graded locally. Other languages need a Judge0 server, which is not configured. |
| Report and share link (E9) | **WORKS** | Covered by the previous Playwright test, which **typed** its answers and never used voice. |
| Design quality | **BROKEN against the brief** | System font, default blue buttons, no design system, no dark mode (screenshots). |
| End-to-end test through voice | **MISSING** | `apps/web/tests/session.e2e.ts` fills a text box, not the microphone. |
| Cross-browser tests (Firefox, WebKit, mobile) | **MISSING** | Only Chromium is installed in this workspace. CI ran Chromium only. |
| Hosted preview (Vercel) | **WORKS, same offline behaviour** | It sits behind Vercel login and is configured identically, so it has the same bank questions, no STT and no TTS. |
| Legacy desktop prototype (`legacy/desktop`) | **BROKEN** | `uv pip compile legacy/desktop/requirements.txt` fails: `mediapipe==0.10.3` cannot be installed on Python 3.11. It also needs a display and a webcam. |
| Legacy "futuristic" prototype (`legacy/futuristic`) | **BROKEN** | It has no requirements file at all, and it is a desktop GUI. |

### Randomness in product code (grep: `random`, `Math.random`)

| Location | Why | Verdict |
|---|---|---|
| `nlp/question_bank.py` | Picks a question from the bank | **Remove.** The bank is deleted in the rebuild. |
| `voice/prompts.py` | Random phrases for voice enrolment | **Keep.** Unpredictable phrases are the anti-replay measure. |
| `codeexec/challenges.py` | Picks a coding challenge (seeded) | **Review** in the rebuild. |
| `components/Avatar.tsx` | Blink interval | **Cosmetic.** Replaced by the 3D avatar. |

### Verdict

The owner's report is accurate:
- **Questions are static.**
- **No transcript is produced.**
- **The interviewer is silent or robotic.**
- **The avatar is a 2D cartoon whose mouth doesn't follow the audio.**
- **The UI is unstyled.**

Several of these pass their unit tests only because they run under offline defaults, and the one end-to-end test types its answers instead of speaking. The rebuild order follows from this: voice pipeline, then question engine, then avatar, then UI.

## Run 2: after the rebuild (2026-10-06, commit `f1a2634`)

**How it was run.** Same questions as Run 1, answered by running the product the way a user would and by the automated runs that do the same, each linked below.
- Stack: `docker compose up -d --build` from empty volumes, with the `.env.example` defaults. The only sandbox-specific addition is a proxy override that is not committed. Log: [`compose-up.log`](evidence/compose/compose-up.log).
- Voice: synthetic WAV answers (Kokoro voices, one of them a Hindi voice speaking English) are played into the browser's microphone. The spoken-interview E2E (`apps/web/tests/interview.e2e.ts`) then drives the real UI from the device check to the report.
- Real LLM: no API key is available here. The CI `full` job runs the same E2E with an open-weights model (Qwen2.5-7B-Instruct on Ollama, CPU only) and `E2E_REQUIRE_LLM=1`, which fails the run on any backup question or any answer left with offline scoring.

### Starting the app

| Feature | Status | Evidence |
|---|---|---|
| One command: `docker compose up --build` | **WORKS** | [`compose-up.log`](evidence/compose/compose-up.log): 3 min 12 s from empty volumes, including the checksum-verified download of the four speech models. The spoken E2E then passed against that stack (3/3, [`e2e/compose-no-llm/`](evidence/e2e/compose-no-llm/)). The Run 1 blockers are fixed: the API image no longer pulls `uv` from ghcr.io, and an interrupted model download now resumes (D20). |
| Native start (`uvicorn` + `next start`) | **WORKS** | This is how the CI E2E jobs start the app (`apps/web/playwright.config.ts`). `/ready` reports `stt: sherpa, tts: kokoro` with 7 voices. |

### Core interview loop (what the owner reported)

| Question asked | Answer | Evidence |
|---|---|---|
| Does question text come from a static list? | **NO. WORKS.** | The question bank is gone. Every question is written live by the LLM interviewer agent from a blueprint, the candidate's parameters and their answers. *(Mock-interview pass rate: pending the final run.)* The real-LLM E2E fails on any backup question, and it passed (*pending: final CI run*). With no LLM configured (the compose default), questions are clearly badged backup questions, never silent. |
| Does the mic actually capture audio? | **YES. WORKS.** | AudioWorklet capture at 16 kHz over the WebSocket. The spoken E2E passes on Chromium, Firefox, WebKit, Edge, mobile Chrome and mobile Safari (CI `browsers` jobs). |
| Does voice activity detection end the turn on its own? | **YES. WORKS.** | Silero VAD on the server ends the turn after the WAV answer finishes. The protocol log shows `stt.final` then `phase thinking` without any click (E2E failure dumps and the `?debug=1` panel). |
| Does speech-to-text return real transcripts? | **YES. WORKS.** | Local sherpa-onnx: Nemotron streaming for live captions, Parakeet for the final transcript. On the synthetic 5-voice set, WER is 2.4% streaming and 0.0% final ([`stt_bench.txt`](evidence/voice/stt_bench.txt)). That set is synthetic speech, not human recordings, so it is not a field accuracy figure. Captions and transcript are visible in `04-transcript.png`. |
| Does text-to-speech play? | **YES. WORKS.** | Kokoro neural TTS (local, 7 voices), streamed sentence by sentence. `tts.start` → `audio started` → `audio ended` events appear in every browser run. |
| Does the avatar's mouth move in sync with the audio? | **YES. WORKS.** | A TalkingHead 3D avatar whose visemes are derived from the audio actually playing (HeadAudio). The compose run counted 729 visemes (`03-barge-in.png`, diagnostics panel). |
| Is the feedback derived from the answer? | **YES. WORKS.** | Each answer is scored by the LLM rubric, and every score cites the candidate's own words. Answers not scored yet show "offline scoring" until the LLM finishes (D21). The real-LLM E2E fails if any answer is left with offline scoring (*pending: final CI run*). |

### Everything else

| Feature | Status | Evidence |
|---|---|---|
| Landing, sign-up, consent, dashboard | **WORKS** | [`evidence/ui/`](evidence/ui/): 12 screens × 3 viewports × dark/light. |
| Interview set-up (role, JD, resume) | **WORKS** | Role, seniority, company and its interview style, JD, resume PDF, skills to probe, interview type, round, difficulty, language, duration and persona (`new-interview--*.jpg`). |
| Adaptive difficulty and follow-ups | **WORKS** | Follow-ups must quote the previous answer and ask about it. Difficulty moves at most one step per turn and never against the score. Both are checked by code in every mock interview (*(pending: final mock-interview run)*). |
| Interview blueprint or plan | **WORKS** | Planned at session creation while the candidate checks devices (D18). Shown in the room's sidebar and in the report's "By skill". |
| Streaming STT with partial captions | **WORKS** | Live captions while the candidate speaks (E2E asserts them). |
| Streaming TTS | **WORKS** | First audio after a sentence, not the whole reply. On the CPU-only sandbox, TTS first audio was p50 2.4 s. |
| Barge-in | **WORKS** | The E2E talks over the interviewer while its audio is playing; playback stops and the turn passes to the candidate. `barge-ins 1` appears in the diagnostics. Detected in the browser and on the server. |
| Mic permission errors, device picker, reconnect | **WORKS** | The "microphone blocked" E2E passes on all browsers (recovery steps shown, typing still works). The device check picks the mic and camera (`01-device-check.png`). Reconnect and resume are covered by `services/api/tests/test_live.py`. |
| Diagnostics panel (`?debug=1`) | **WORKS** | Connection, RTT, VAD, mic level, avatar fps, visemes, barge-ins, and per-stage p50/p95 latency (`03-barge-in.png`). |
| 3D avatar, personas, idle/listening/thinking states | **WORKS** | 4 interviewer personas (Maya, Emma, Priya, Ananya), each with its own look and neural voice (7 voices available). Quality tiers with an audio-only fallback. |
| Candidate-side lip-sync verification (patent E4) | **WORKS** | Audio-visual correlation per answer, from the realtime audio and the browser's mouth series. A mismatch raises an integrity notice (`test_lipsync_mismatch_raises_integrity_notice`). |
| Face and voice verification (E2, E3) | **WORKS once a model is configured; off in the default config** | The code and tests are in place (`interview_core.face`, `interview_core.voice`). They need a commercially licensed face-embedding model and a speaker model, which are owner decisions (licence and cost). `/ready` shows `face_verification: false, voice_verification: false` until then. |
| Gaze and integrity notices (E5, E7) | **WORKS** | On-device MediaPipe. With the fake camera's test pattern, "We can't see your face" is correct. Phone detection runs in the same loop. |
| Coding challenge (E8) | **WORKS for SQL; other languages need Judge0** | SQL is graded locally with hidden tests. Python and other languages need a Judge0 server (`code_execution: false` without one). |
| Report and share link (E9) | **WORKS** | The spoken E2E ends on the report (`05-report.png`). Share links are revocable and hide the transcript (`test_full_session_with_signals_and_report`). |
| Design quality | **WORKS** | [`docs/DESIGN.md`](DESIGN.md) design system, dark and light, WCAG AA contrast. Lighthouse mobile: landing 94 / 100, report 98 / 100 (performance / accessibility; [`evidence/lighthouse/`](evidence/lighthouse/)). |
| End-to-end test through voice | **WORKS** | `interview.e2e.ts` speaks into the microphone. The old typed-answer test is gone. |
| Cross-browser tests (Firefox, WebKit, mobile) | **WORKS** | CI `browsers` matrix: Firefox, WebKit, Edge, mobile Chrome (Pixel 7) and mobile Safari (iPhone 14), each with a virtual audio device (*pending: final CI run*). |
| Hosted preview (Vercel) | **NOT LIVE: needs a realtime host (owner decision)** | The web app's preview builds, but Vercel functions cannot host the realtime speech service (WebSocket, speech models, LLM). Going live needs a container host and an LLM key, chosen and paid for by the owner. |
| Legacy desktop prototypes (`legacy/`) | **Kept as reference, not run** | Their algorithms are preserved in `interview_core.legacy` and pinned by characterization tests against the original code. |

### Randomness in product code (re-run: [`evidence/audit/no-mocks.md`](evidence/audit/no-mocks.md))

| Location | Why | Verdict |
|---|---|---|
| Question selection | The bank is deleted; questions come from the LLM | **Removed** |
| `voice/prompts.py`, `face/liveness.py` | Unpredictable enrolment phrases and liveness steps | **Keep.** Anti-replay. |
| `codeexec/challenges.py` | Tie-break among equally suitable challenges, seeded by session id | **Keep.** Deterministic per session; not scoring. |
| `lib/voice/realtime.ts` | Reconnect backoff jitter | **Keep.** Networking only. |
| Avatar blinking | TalkingHead's own animation | Cosmetic, in a vendored library |

### Tests on this commit

- Python: 180 passed (core, API and eval, including the live speech-model tests; 96 s on 3e0f5b1).
- Web unit: 13 passed.
- Spoken E2E: 3 per browser on 6 browser projects in CI, plus the compose run.
- Real-LLM E2E: *pending: final CI run*.
- Mock interviews with the real LLM: *(pending: final mock-interview run)*.

### Verdict

Every item the owner reported is fixed and observed working:
- **Questions** are generated live by an LLM, not taken from a list.
- **Transcripts** are real.
- **The interviewer** speaks with a neural voice.
- **The avatar** is 3D and lip-synced to the audio it plays.
- **The UI** follows a design system.

Two items are not WORKS by default, and both need an owner decision rather than code:
- **Face and voice verification** need licensed models.
- **Hosting** needs a server for the realtime service and an LLM key.

Accuracy is still not measured on real users. No consented evaluation data exists (`docs/EVAL_REPORT.md`), so every score in the product is labelled "experimental".
