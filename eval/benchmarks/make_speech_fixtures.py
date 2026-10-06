"""Generate the synthetic spoken-answer fixtures used by the web E2E tests (see apps/web/tests/fixtures/speech/README.md).

Usage: MODELS_DIR=... uv run python eval/benchmarks/make_speech_fixtures.py [out_dir]
"""

import sys
import wave

import numpy as np
from interview_core.realtime.tts import KokoroTTS

out = sys.argv[1] if len(sys.argv) > 1 else "apps/web/tests/fixtures/speech"
answers = {
    "answer-1-us": (
        "maya",
        "In my last role I owned the billing service. The nightly batch job took six hours, so I profiled it, found that most of the time went into one unindexed join, added a composite index and rewrote the job to stream rows. The run dropped to forty minutes, and I verified it by comparing a week of outputs before and after.",
    ),
    "answer-2-in": (
        "priya",
        "Sure. We had two teams that disagreed about the API contract. I set up a short design review, wrote down both proposals with their trade offs, and we agreed on versioned endpoints. After that, integration bugs went down a lot and releases became weekly instead of monthly.",
    ),
    "answer-3-uk": (
        "emma",
        "Honestly I am not completely sure. I think I would start by measuring the latency at each hop, and then look at caching the most frequent reads, but I have not done that at a large scale before.",
    ),
    "answer-4-in": (
        "ananya",
        "I would like to know how the team measures success in the first ninety days, and what the on call rotation looks like.",
    ),
}
tts = KokoroTTS()
for name, (voice, text) in answers.items():
    pcm = b"".join(c.pcm16 for c in tts.synthesize(text, voice))
    x = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768
    y = np.concatenate([np.zeros(6000, np.float32), x, np.zeros(12000, np.float32)])
    with wave.open(f"{out}/{name}.wav", "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(24000)
        w.writeframes((np.clip(y, -1, 1) * 32767).astype("<i2").tobytes())
    print(name, round(len(y) / 24000, 1), "s")
