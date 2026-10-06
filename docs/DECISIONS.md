# Decisions log

One entry per decision: what was chosen, the alternatives, the evidence, and how to reverse it.
Newest decisions are at the bottom. "Verified" means checked against a primary source (repo, licence file, package tarball, or a run in this workspace); anything else is labelled.

---

## D1. Local speech-to-text: Nemotron streaming 0.6B for live captions, Parakeet TDT 0.6B v2 for final text (2026-10-06)

**Chosen:** both run through `sherpa-onnx` (Apache-2.0, PyPI). Models come from the sherpa-onnx GitHub releases.

**Benchmark.** Run in this workspace on 4 CPU threads, with audio fed in 100 ms chunks as a browser would send it. Script and output: `eval/benchmarks/stt_bench.py` and `docs/evidence/voice/stt_bench.txt`. The test set is 30 synthetic utterances: 6 interview answers in 5 Kokoro voices, two of them Hindi voices speaking English.

| Model | Mode | WER | Real-time factor | First partial |
|---|---|---|---|---|
| zipformer-en-2023-06-26 (LibriSpeech) | streaming | 12.7% | 0.14 | 0.80 s |
| NeMo FastConformer streaming 80 ms | streaming | 6.7% | 0.83 | 0.60 s |
| **Nemotron speech streaming en 0.6B, 560 ms, int8** | streaming | **2.4%** | 0.33 | 0.70 s |
| **Parakeet TDT 0.6B v2, int8** | offline final pass | **0.0%** | 0.074 | — |

**Caveat, stated plainly:** synthetic speech is far easier than real candidates. These numbers rank the models; they are not accuracy claims. Real accuracy, including Indian-English accents, needs the consented recordings in `NEEDS_FROM_KASHYAP.md` item 6.

**Design.** Partial captions come from the streaming model. Each speech segment (split on pauses by Silero VAD) is re-decoded by Parakeet as soon as it closes, so when the turn ends only the last segment is left to process.

**Licences.** These are from the upstream model cards and were *not* re-verified, because huggingface.co is blocked here:
- Nemotron: NVIDIA Open Model License, commercial use permitted.
- Parakeet TDT 0.6B v2: CC-BY-4.0, so attribution is required (in `MODEL_CARDS/`).

**Premium option.** Deepgram Nova-3 streaming, selected with `STT_PROVIDER=deepgram`. It can't be tested here because `api.deepgram.com` is blocked.

## D2. Local text-to-speech: Kokoro v1.0 through sherpa-onnx (2026-10-06)

**Chosen:** Kokoro-82M v1.0. Weights are Apache-2.0 (verified: the `LICENSE` file is in the model package).

**Speed.** Real-time factor 0.28 median, 0.55 worst case, on 4 threads (measured while building the test set). A 3-second sentence is ready in about 0.8 s.

**No timestamps.** sherpa-onnx text-to-speech returns no word or phoneme timestamps. This is verified in the source (`GeneratedAudio` has only samples and sample_rate). Lip-sync is therefore driven from the audio itself (see D4).

**Licence caveat.** sherpa-onnx's text-to-speech links espeak-ng (GPL-3.0) for phonemisation. Running it on our server is fine. Distributing a Docker image that contains it means complying with the GPL for that component. This is flagged for the lawyer in `COMPLIANCE.md`.

**Premium options.** ElevenLabs (character timestamps) and Amazon Polly (viseme marks), selected with `TTS_PROVIDER`.

**Rejected:**
- Piper `en_US-lessac` (Blizzard licence, restrictive).
- HeadTTS. Its timestamped Kokoro build is hosted on Hugging Face, which is unreachable here.

## D3. Local LLM: none bundled. Hosted API by default, Ollama optional (2026-10-06)

**Finding (verified):** no first-party small instruct model (Qwen, Llama, Phi, Gemma) is published on GitHub, npm or PyPI. The only GitHub copy found is an unofficial Qwen3-0.6B safetensors mirror with unverifiable provenance, so it was rejected.

**Chosen:**
- An OpenAI-compatible provider covers Ollama, vLLM, LM Studio, Groq, OpenRouter and Together.
- Native providers for Anthropic and Gemini.
- `docker compose --profile local-llm` starts Ollama on the owner's machine.
- Real LLM evidence is produced by a GitHub Actions job running Ollama, because GitHub runners can reach ollama.com. Once a key is added, it is also produced here with that provider.

## D4. Avatar: TalkingHead 1.7.0 on three.js 0.180, MPFB (CC0) character, audio-driven lip-sync with HeadAudio (2026-10-06)

**Verified facts** (from the subagent's research, with repo clones and GLB inspection):
- TalkingHead is MIT and supports ARKit 52 blendshapes plus Oculus visemes, blinking, eye contact, idle motion, moods and gestures.
- Of its sample avatars, only `mpfb.glb` is commercially usable (CC0).
- The others are not commercially usable:
  - the Ready Player Me sample is CC BY-NC;
  - the Avaturn, AvatarSDK and VRoid samples are licensed "for non-commercial use".
- Ready Player Me shut down its public platform on 2026-01-31.

**Lip-sync.** HeadAudio (MIT; its 14 KB model is inside the npm package) turns the audio that is actually playing into Oculus viseme weights in an AudioWorklet. It works with any text-to-speech provider, including Kokoro, which has no timestamps. When a provider does return timings (Polly visemes, ElevenLabs characters), they are used for the timeline instead.

**Photoreal option (benchmark recorded later in this file):** Anam, Tavus, HeyGen LiveAvatar and Simli all accept audio from our own text-to-speech. All need paid keys, and none is reachable from this workspace.

**Rejected:**
- Ready Player Me (shut down, and its sample is non-commercial).
- Avaturn (commercial terms unverified, PRO plan $800 a month).
- MetaHuman (no web or GLB pipeline).
- Rocketbox (older low-poly FBX that would need re-rigging).
- NVIDIA Audio2Face-3D (needs an NVIDIA GPU; models on Hugging Face).

## D5. Dockerfiles must not depend on ghcr.io (2026-10-06)

The truth run showed that `COPY --from=ghcr.io/astral-sh/uv` fails on networks that block ghcr.io. Images now install `uv` from PyPI instead, which works wherever pip works.

## D6. Personas are female-presenting until a second avatar model exists (2026-10-06)

There is exactly one commercially usable rigged avatar with ARKit blendshapes (the MPFB CC0 model in D4), and it is female. Pairing a male voice with it looks wrong, so the four personas are Maya (US), Emma (UK), Priya and Ananya (Indian English voices). The male Kokoro and Polly voices stay in the voice tables. Adding male personas needs a second model exported from MPFB (Blender + MPFB add-on, CC0) and run through `apps/web/scripts/optimize-avatar.sh`. That is a follow-up, listed in `NEEDS_FROM_KASHYAP.md` only if a commissioned or paid avatar is wanted instead.

## D7. Avatar instances mount into their own node (2026-10-06)

React mounts effects twice in development. The first, cancelled TalkingHead instance used to tear down the shared container when it finished loading, which removed the second instance's canvas and left the stage empty. Each instance now renders into its own child element.

## D8. Schema-constrained replies for local and Gemini models; grounded anchor quotes (2026-10-06)

**Measured** (first CI evidence run, Qwen2.5-7B-Instruct on a 4-vCPU GitHub runner, case mock-03): calls took 1–4 minutes. Of the first 12 turns, 6 ended in the flagged emergency question. The causes were missing `<plan>` tags, invalid plan JSON, paraphrased `anchor_quote`s, an out-of-range competency or difficulty, and read timeouts. The harness then crashed because `AgentState.from_dict` mutated its input.

**Chosen:**
- Providers that support it (OpenAI-compatible endpoints, including Ollama ≥ 0.5 and vLLM, plus Gemini) now get a JSON schema for every call. The schema covers the blueprint, and for each turn: the action (only the forced one when timing forces it), competency ids, difficulty 1–5, the assessment and `say`. The decoder can then only produce valid structure.
- Anthropic keeps the tagged format, and the agent accepts both.
- A follow-up's quote that is not verbatim is replaced with the candidate's real words it overlaps (at least half of the quote's words, at least three), logged as a correction. Otherwise the reply is still rejected. The stored anchor is always verbatim, which is stricter than before: an 80% bag-of-words match used to be stored as the model wrote it.
- The evidence job runs one interview per job (20 jobs), because one 20-minute simulated interview takes up to an hour on CPU.
- `from_dict` deep-copies its input.

## D9. Interviewer playback is tracked per utterance in the client (2026-10-06)

**Measured** (local E2E, protocol event log in the `?debug=1` panel): TalkingHead's stream callbacks fire once per stream session, not per utterance. Its "ended" callback also fired about 4 s into an 8.1 s question when synthesis fell behind playback. And `streamAudio` transfers (detaches) the PCM buffer, so measuring it afterwards read 0 bytes. Together these made the room start listening while the interviewer was still talking, and disabled barge-in after the first question.

**Chosen:**
- The Speaker counts queued audio before handing it over.
- It reports "started" on the first chunk of each utterance.
- It reports "ended" only after the server's `tts.end` and once the queued duration has played. The library callback can confirm the end but never shortens it.
- Barge-in uses Silero VAD, plus an energy fast path: 400 ms at or above about -32 dBFS while the interviewer's audio is playing. The fast path covers moments when VAD inference lags on a busy main thread.
- A `software` quality tier (10 fps, half resolution, lighter vision sampling) applies when WebGL runs on the CPU (SwiftShader, llvmpipe).

## D10. E2E speech goes through an injected fake microphone (2026-10-06)

Chrome's `--use-fake-device-for-media-stream` flags exist only in Chromium and can't be timed to the conversation. The E2E suite instead replaces `getUserMedia` with a Web Audio destination plus a canvas camera, and plays recorded WAVs into it when the interviewer is listening. Everything after the microphone is real: the AudioWorklet, the WebSocket, the server VAD, sherpa-onnx recognition, the agent, Kokoro and the avatar.

The WAVs are synthetic (Kokoro voices, including two Hindi voices speaking English) and labelled as such in `apps/web/tests/fixtures/speech/README.md`. Real recorded voices are a request in `NEEDS_FROM_KASHYAP.md`.

The same test runs on Chromium (CI web job), and on Firefox, WebKit, Edge and the Pixel 7 and iPhone 14 viewports (`e2e.yml`). It runs once more with a real LLM, where any backup question fails the run (`e2e.yml`, job `full`).

## D11. Agent repairs are grammar-constrained by the failure (2026-10-06)

**Measured** (agent-evidence run 3, Qwen2.5-7B on CPU, after D8): calls dropped to 26–58 s. In case 3 all 10 follow-ups quoted the candidate correctly and difficulty tracked performance. Three turns still fell back to the backup question: the model kept re-asking a generic "could you give more details about the specific tools…" or quoting words that were not in the answer, even after one repair.

**Chosen:** two repairs instead of one. On schema-capable providers, the repair narrows the grammar according to what failed:
- an unverifiable `anchor_quote` may only be one of the answer's own clauses;
- a repeated question must move on (`new_topic` or `revisit`) to a different competency.

Timing-forced actions (`open`, `wrap_up`, `close`) are never overridden. The checks themselves are unchanged.

## D12. Docker builds behind TLS-intercepting proxies (2026-10-06)

Both Dockerfiles accept an optional BuildKit secret `extra_ca` (`--secret id=extra_ca,src=ca.pem`) used only during dependency installation. Without it, builds behave exactly as before. The API image no longer runs `apt-get`: the speech runtime works on `python:3.11-slim` as is, which was verified by running the full E2E against the compose stack.

## D13. Barge-in is detected on the server too; the socket never blocks on the LLM (2026-10-06)

**Measured** (CI, cd12703):
- On 4-vCPU runners with software WebGL, the browser's VAD fired only after the question audio had ended (Chromium and Edge failed the barge-in check; WebKit passed).
- In the real-LLM job (Qwen2.5-7B on CPU, 60–130 s for the first turn), the connection dropped 3 times. The receive loop was awaiting the turn, mic frames went unread, and pings timed out. Each reconnect started another planning call.
- SQLite reported "database is locked" because the turn's transaction stayed open during the LLM call.

**Chosen:**
1. The server runs a Silero gate (sherpa-onnx) on the mic stream while the interviewer's audio is playing. On sustained speech (≥ 0.3 s, p ≥ 0.6) it cancels the question and starts the turn with the last 2.5 s of audio, so the interrupting words are kept. The same pre-roll is used when the browser detects the barge-in. Set `SERVER_BARGE_IN=0` to disable.
2. Commands that can wait on the LLM run on a per-connection consumer task. The receive loop only reads frames and handles quick messages. A turn that is in flight survives a dropped connection, and the reconnect waits for it instead of generating a second one.
3. `advance()` commits before calling the LLM and writes the turn in a new transaction.
4. The client never awaits `AudioContext.resume()` when joining. Without an audio device, for example in headless Firefox, it can stay pending forever.

Tests: `test_server_detects_barge_in_and_keeps_the_interrupting_words` and `test_socket_stays_responsive_while_a_slow_llm_thinks` (real speech models).

## D14. Repair prompts carry a concrete next step (2026-10-06)

**Measured** (agent-evidence run 4, after D11): cases that failed had exactly one backup question each. The rejected attempts were mostly a repeated drill-down (asking again for metrics the candidate had already said they lacked) and replies that asked no question.

**Chosen:**
- After a repeat, the repair message names the least-covered other competency (its id, name and why) to move to.
- After a reply with no question, the repair message asks for exactly one direct question ending in "?".
- The evidence harness now reports counts of rejection reasons for each case.
- Superseded evidence and E2E runs are cancelled automatically.

## D15. HeadAudio is re-wired whenever TalkingHead rebuilds its audio graph (2026-10-06)

**Measured** (new viseme counter in `?debug=1`, chromium E2E against `docker compose up`): 0 audio-driven viseme updates during interviewer speech. The cause: `streamStart({sampleRate: 24000})` makes TalkingHead call `initAudioGraph(24000)` whenever the context runs at a different rate, which closes the AudioContext and recreates every node. HeadAudio, and the analyser used for the speaking ring, stayed attached to the closed graph, so **the avatar's mouth never followed the audio** and the ring read zero.

**Chosen:**
- The graph is created at the TTS rate (24 kHz) before wiring.
- HeadAudio is re-wired after any `streamStart` that rebuilds the context (for other providers' rates).
- Level and diagnostics always read the current analyser.

**After the fix:** viseme updates with five distinct viseme shapes and a peak weight of 0.75 in the same run. The E2E now asserts `visemes > 0` on Chromium.

## D16. Grounding a paraphrased anchor, per-attempt retries, and a timeout that fits self-hosted models (2026-10-06)

**Measured.** Agent-evidence run 5 (48e3da0, Qwen2.5-7B on CPU) and the real-LLM E2E on e30b833.

In interview 02, 7 of 19 attempts were rejected:
- 5 for "anchor_quote must be copied verbatim";
- 2 for repeating an earlier question;
- 1 for an Ollama `ReadTimeout` (240 s).

The one backup question in that interview came from a turn where the first reply was rejected and the repair then timed out. Two code defects made a single timeout fatal:
1. **Shared retry counter.** Transient-error retries were counted across the whole turn, so a timeout during a repair was never retried.
2. **Fixed 30 s timeout.** The API built the agent with a 30 s request timeout. A CPU-hosted 7B model can take longer than that before its first token on a 3–4k-token prompt. That is the likely cause of the one backup question in the real-LLM E2E run.

**Chosen:**
- **Second grounding pass.** If `ground_quote` cannot find the model's quote in the candidate's last answer, `ground_by_topic` picks the answer clause that shares the most content words with the quote and the question together. It accepts that clause only when at least 3 distinct content words are shared.
  - The stored anchor is still the candidate's own words, verbatim.
  - A follow-up that refers to nothing the candidate said is still rejected and repaired. The follow-up mechanism keeps its check; the check now verifies what the question is about instead of failing on the model's copying accuracy.
- **Per-attempt retries.** Transient retries are counted per attempt, so a timeout during a repair gets its own retry.
- **Configurable timeout.** `LLM_TIMEOUT_S` sets the request timeout. When it is not set, the default is 30 s for hosted APIs and 240 s when the chain contains `ollama` or `openai_compat`.

Tests:
- `test_paraphrased_quote_is_grounded_to_the_clause_the_question_is_about`
- `test_transient_error_during_a_repair_still_gets_its_retry`
- `test_llm_timeout_defaults`

## D17. Playback end follows the audio actually played; the avatar sheds load on a starved device (2026-10-06)

**Measured.** Local E2E pinned to 2 cores (`taskset -c 0,1`, about the size of a CI runner):
- **TTS slower than real time.** Kokoro needed 17.7 s to synthesise 8.1 s of speech, so playback stalls between chunks.
- **Starved main thread.** Software WebGL ran at about 1 fps, and React rendered the transcript about 15 s late. The client also handled WebSocket messages late.
- **Fallback timer fired early.** The server's fallback timer for "the client never reported playback end" was `tts.end + duration + 3 s`. It fired while the lagging client was still playing, so the server switched to listening before the client reported "audio ended" (protocol trace: `phase listening` 48.6 s, `audio ended` 50.1 s).
- **Client estimate ignored stalls.** The client's own end estimate, `first chunk + total queued audio`, assumed no stalls either.

**Chosen:**
1. **Play-out model and fallback timer.**
   - The speaker models play-out per chunk: each chunk starts when it arrives or when the previous chunk ends, whichever is later. "Ended" is reported only after the modelled end.
   - The server's fallback timer allows `duration + max(5 s, duration / 2)` after `tts.end`. It covers clients that never report; a client that is merely slow is no longer cut off.
2. **Adaptive avatar quality.**
   - Trigger: the avatar holds less than half its target frame rate for 3 s.
   - Response: it renders smaller frames (pixel ratio × 0.6) at a lower rate (fps × 0.6, minimum 5), at most twice.
   - Why: audio capture, barge-in detection and the UI share the main thread, and they come first.
   - Visibility: `?debug=1` shows the degradation level.
3. **E2E barge-in check.**
   - Old method: waiting for the stage's `data-audio` attribute, which a starved page renders late.
   - New method: wait on the room controller's state.
   - Diagnostics: a failing test prints the protocol trace and the browser console into the job log.
4. **Fake microphone and browser setup.**
   - The fake microphone is installed with `Object.defineProperty` on `MediaDevices.prototype` and on the instance. WebKit ignores plain assignment and hands the page its own mock devices.
   - The fake microphone fails with a clear message instead of hanging when the browser has no audio device.
   - CI gives Firefox and WebKit a PulseAudio null sink.
5. **Playback worklet preloaded.**
   - Problem: on the starved device, TalkingHead's own 5 s timeout for loading its playback worklet fired ("Worklet loading timed out"), and the interviewer's audio stream failed to start.
   - Fix: the worklet is now loaded without a timeout while the avatar loads. If the library's own load still fails, the stream is retried once.
6. **Lip-sync counter measures detection, not rendering.**
   - The `?debug=1` viseme counter counts the non-silence visemes HeadAudio detects in the played audio, not rendered frames, which a starved device draws at 1–2 fps.
   - Barge-ins are counted once per utterance, even when both the browser and the server detect them.

**After:** the full spoken-interview E2E passes on the same 2-core pin (barge-in, lip-sync, report), where it previously failed at barge-in and then at lip-sync.
