"""Generate cross-language parity fixtures for apps/web/lib/visionMath.ts and visemes.ts.

Run: uv run python tools/gen_parity_fixtures.py. packages/core/tests/test_parity_fixtures.py
fails if the committed fixture no longer matches the Python implementation.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from interview_core.avatar.visemes import from_text
from interview_core.gaze import extract_features, is_on_screen
from interview_core.lipsync import mouth_aperture

OUT = Path(__file__).resolve().parents[1] / "apps/web/lib/__fixtures__/parity.json"


def build() -> dict:
    rng = np.random.default_rng(42)
    cases = []
    for _ in range(25):
        lm = rng.uniform(0.2, 0.8, (478, 2)).round(5)
        f = extract_features(lm)
        cases.append(
            {
                "landmarks": lm.tolist(),
                "mouth": mouth_aperture(lm),
                "gaze": {"horizontal": f.horizontal, "vertical": f.vertical, "yaw": f.yaw, "pitch": f.pitch},
                "on_screen": is_on_screen(f),
            }
        )
    texts = ["Hello world", "Tell me about a time you led a team.", "", "Why?"]
    return {
        "landmarks": cases,
        "visemes": [{"text": t, "duration": 1200, "timeline": from_text(t, 1200)} for t in texts],
    }


if __name__ == "__main__":
    OUT.write_text(json.dumps(build()), encoding="utf-8")
    print(f"wrote {OUT}")
