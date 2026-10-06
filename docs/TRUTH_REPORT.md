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

## Run 2: after the rebuild

*Re-run at the end of the rebuild, using the same method.*
