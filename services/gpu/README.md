# GPU services (optional)

Nothing on the interview's critical path needs a GPU: the avatar is an in-browser viseme rig, face and gaze signals are computed on the device, and ASR/TTS can be vendor APIs. Two optional workloads may run here.

## 1. Neural talking-head avatar (`FEATURE_NEURAL_AVATAR`)

**Contract** (called by `services/api` → `POST /avatar/render`, 1.5 s timeout):

```
POST {NEURAL_AVATAR_URL}/render
{"text": "Tell me about a time you led a team."}
→ 200 {"stream_url": "https://.../clip.m3u8", "gpu_seconds": 0.8}
```

- **Model licence:** must be commercially licensed. Wav2Lip, SadTalker-class research weights and anything trained on LRS2/LRS3 are excluded (see `docs/PLAN.md` §f). Options are a vendor with a commercial licence, or a model trained on consented, commercially licensed footage.
- **Latency budget:** the browser waits at most 1.2 s for `stream_url`, otherwise it uses the rig. Stream the first frames (HLS/WebRTC); don't render whole clips before responding.
- **Hosting:** a scale-to-zero GPU platform (Modal or RunPod Serverless). Keep one warm instance during business hours if p95 first-frame time matters more than idle cost. Cold starts only ever degrade to the rig, never block a session.
- **Evaluation:** LSE-D / LSE-C via the `avatar_sync` suite, plus `question_to_first_avatar_frame` latency.

## 2. Batch ASR fallback

`interview_core.speech.asr.FasterWhisperASR` (MIT weights) can run on a GPU worker for re-transcription and evaluation runs. Set `ASR_PROVIDER=faster_whisper` on the worker.
