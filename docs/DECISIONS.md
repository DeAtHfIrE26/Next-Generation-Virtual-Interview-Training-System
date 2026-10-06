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
