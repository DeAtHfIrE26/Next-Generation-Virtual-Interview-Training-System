"""STT benchmark (docs/DECISIONS.md D1). Synthetic interview answers spoken by Kokoro voices.

Env: BENCH_DIR (audio + manifest.json, default eval/benchmarks/out), BENCH_MODELS (sherpa-onnx model dirs).
Results of the run cited in DECISIONS.md: docs/evidence/voice/stt_bench.txt
"""

import os

BENCH_DIR = os.environ.get("BENCH_DIR", "eval/benchmarks/out")
BENCH_MODELS = os.environ.get("BENCH_MODELS", ".models/bench")
# Synthetic STT test set: interview-style answers in 5 Kokoro voices (incl. two Hindi voices speaking English) + Piper.
import json
import time

import numpy as np
import sherpa_onnx
import soundfile as sf

M = os.path.join(BENCH_MODELS, "kokoro-multi-lang-v1_0")
cfg = sherpa_onnx.OfflineTtsConfig(
    model=sherpa_onnx.OfflineTtsModelConfig(
        kokoro=sherpa_onnx.OfflineTtsKokoroModelConfig(
            model=f"{M}/model.onnx",
            voices=f"{M}/voices.bin",
            tokens=f"{M}/tokens.txt",
            data_dir=f"{M}/espeak-ng-data",
            lexicon=f"{M}/lexicon-us-en.txt",
        ),
        num_threads=4,
    )
)
tts = sherpa_onnx.OfflineTts(cfg)
texts = [
    "In my last role I owned the payments reconciliation service and reduced the failure rate from four percent to under half a percent.",
    "I would start by clarifying the requirements, then sketch the data model, and only after that think about caching and sharding.",
    "We had a disagreement about the release date, so I set up a short meeting, listed the risks, and we agreed to ship a smaller scope first.",
    "Honestly I have not used Kubernetes in production, but I have deployed containers on a managed service and I understand the basics.",
    "The bottleneck was the database, so I added a read replica and moved the reporting queries off the primary.",
    "My biggest weakness is that I sometimes take on too much myself, so now I break work into tickets and delegate earlier.",
]
voices = {"af_heart": 3, "am_michael": 16, "bf_emma": 21, "hf_alpha": 31, "hm_omega": 33}
manifest = []
for vn, sid in voices.items():
    for i, t in enumerate(texts):
        t0 = time.time()
        a = tts.generate(t, sid=sid, speed=1.0)
        dt = time.time() - t0
        x = np.asarray(a.samples, np.float32)
        n = int(len(x) * 16000 / a.sample_rate)
        y = np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x).astype(np.float32)
        p = f"{BENCH_DIR}/{vn}_{i}.wav"
        sf.write(p, y, 16000, subtype="PCM_16")
        manifest.append(
            {
                "path": p,
                "text": t,
                "voice": vn,
                "dur": len(y) / 16000,
                "tts_rtf": dt / (len(x) / a.sample_rate),
            }
        )
json.dump(manifest, open(os.path.join(BENCH_DIR, "manifest.json"), "w"), indent=1)
r = [m["tts_rtf"] for m in manifest]
print(
    "kokoro utterances",
    len(manifest),
    "TTS RTF median",
    round(sorted(r)[len(r) // 2], 3),
    "max",
    round(max(r), 3),
)
