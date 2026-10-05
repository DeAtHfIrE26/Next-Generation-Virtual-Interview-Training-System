"""Synthetic data for SMOKE runs only.

These generators exist to prove each pipeline executes end to end. Nothing produced from
them may be reported as accuracy; the report renders smoke runs without metric values.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from eval_harness.io import write_wav_mono

CONSENT = "synthetic-smoke"


def syllable_envelope(rng: np.random.Generator, seconds: float, rate_hz: float = 100.0) -> np.ndarray:
    """Smoothed on/off bursts at roughly 4 syllables per second, sampled at ``rate_hz``."""
    n = int(seconds * rate_hz)
    env = np.zeros(n)
    t = 0
    while t < n:
        dur = int(rng.uniform(0.12, 0.3) * rate_hz)
        gap = int(rng.uniform(0.05, 0.25) * rate_hz)
        env[t : t + dur] = rng.uniform(0.5, 1.0)
        t += dur + gap
    k = np.hanning(9)
    return np.convolve(env, k / k.sum(), mode="same")


def speech_like_audio(env_100hz: np.ndarray, sr: int, rng: np.random.Generator) -> np.ndarray:
    n = int(len(env_100hz) * sr / 100)
    env = np.interp(np.arange(n), np.arange(len(env_100hz)) * sr / 100, env_100hz)
    carrier = rng.normal(0, 1, n) * 0.3 + np.sin(2 * np.pi * 140 * np.arange(n) / sr)
    return (0.3 * env * carrier).astype(np.float32)


def aperture_series(env_100hz: np.ndarray, fps: int, lag_s: float, rng: np.random.Generator) -> list[float]:
    shift = int(lag_s * 100)
    lagged = np.roll(env_100hz, shift)
    idx = np.arange(0, len(env_100hz), 100 / fps).astype(int)
    ap = 0.005 + 0.06 * lagged[idx] + rng.normal(0, 0.003, len(idx))
    return np.clip(ap, 0, None).tolist()


def _write_manifest(d: Path, items: list[dict]) -> None:
    d.mkdir(parents=True, exist_ok=True)
    (d / "manifest.jsonl").write_text("\n".join(json.dumps(i) for i in items) + "\n", encoding="utf-8")


def make_face(root: Path, rng: np.random.Generator) -> None:
    import cv2

    d = root / "face_verification"
    (d / "img").mkdir(parents=True, exist_ok=True)
    items = []
    for s in range(4):
        base = rng.integers(0, 256, (64, 64)).astype(np.float32)
        for k in range(5):
            img = np.clip(base + rng.normal(0, 20, base.shape), 0, 255).astype(np.uint8)
            name = f"img/s{s}_{k}.png"
            cv2.imwrite(str(d / name), img)
            items.append(
                {
                    "id": f"s{s}_{k}",
                    "consent_id": CONSENT,
                    "subject_id": f"s{s}",
                    "role": "enroll" if k < 3 else "probe",
                    "image": name,
                    "subgroups": {"lighting": "a" if k % 2 else "b"},
                }
            )
    _write_manifest(d, items)


def make_lipsync(root: Path, rng: np.random.Generator) -> None:
    d = root / "lipsync"
    (d / "clips").mkdir(parents=True, exist_ok=True)
    sr, fps, items = 16000, 30, []
    for i in range(6):
        env = syllable_envelope(rng, 6.0)
        mismatch = i % 2 == 1
        mouth_env = syllable_envelope(rng, 6.0) if mismatch else env
        write_wav_mono(d / f"clips/{i}.wav", speech_like_audio(env, sr, rng), sr)
        series = {
            "fps": fps,
            "width": 640,
            "height": 480,
            "aperture": aperture_series(mouth_env, fps, 0.04, rng),
        }
        (d / f"clips/{i}.json").write_text(json.dumps(series), encoding="utf-8")
        items.append(
            {
                "id": f"c{i}",
                "consent_id": CONSENT,
                "subject_id": f"s{i // 2}",
                "audio": f"clips/{i}.wav",
                "mouth_series": f"clips/{i}.json",
                "label": "mismatch" if mismatch else "genuine",
                **({"mismatch_kind": "other_speaker_audio"} if mismatch else {}),
            }
        )
    _write_manifest(d, items)


def make_gaze(root: Path, rng: np.random.Generator) -> None:
    d = root / "gaze"
    d.mkdir(parents=True, exist_ok=True)
    frames = [
        {"landmarks": rng.uniform(0.05, 0.95, (478, 2)).tolist(), "label_on_screen": bool(rng.integers(0, 2))}
        for _ in range(10)
    ]
    (d / "clip.json").write_text(
        json.dumps({"width": 640, "height": 480, "frames": frames}), encoding="utf-8"
    )
    _write_manifest(d, [{"id": "g0", "consent_id": CONSENT, "subject_id": "s0", "frames": "clip.json"}])


def make_answers(root: Path, rng: np.random.Generator) -> None:
    answers = [
        "First I identified the bottleneck, then we profiled the database queries and finally added an index.",
        "I am not sure, maybe I would ask someone.",
        "Our team collaborated on a cross-functional project; for example we designed an API together.",
        "um I think basically it depends",
        "I would define the problem, design a solution, test it and evaluate the result with users.",
        "I led the migration to the cloud, which reduced cost because we removed idle servers.",
    ]
    items = [
        {
            "id": f"a{i}",
            "consent_id": CONSENT,
            "question": "Tell me about a problem you solved.",
            "answer": a,
            "role": "Software Engineer",
            "human_scores": [int(x) for x in rng.integers(1, 6, 3)],
        }
        for i, a in enumerate(answers)
    ]
    _write_manifest(root / "answer_scoring", items)


def make_runtime(root: Path, rng: np.random.Generator) -> None:
    _write_manifest(
        root / "latency",
        [
            {
                "id": f"l{i}",
                "consent_id": CONSENT,
                "metric": "question_to_first_avatar_frame",
                "ms": float(rng.uniform(200, 900)),
            }
            for i in range(20)
        ],
    )
    _write_manifest(
        root / "llm_schema_validity",
        [
            {
                "id": f"q{i}",
                "consent_id": CONSENT,
                "provider": "smoke",
                "task": "question",
                "raw_valid": bool(i % 5),
                "delivered_valid": True,
                "used_fallback": i % 5 == 0,
            }
            for i in range(10)
        ],
    )
    _write_manifest(
        root / "device_detection",
        [
            {
                "id": f"d{i}",
                "consent_id": CONSENT,
                "detector": "smoke",
                "event": "phone",
                "truth": bool(i % 2),
                "predicted": bool(i % 3),
            }
            for i in range(12)
        ],
    )


def generate(root: Path, seed: int = 0) -> Path:
    rng = np.random.default_rng(seed)
    for make in (make_face, make_lipsync, make_gaze, make_answers, make_runtime):
        make(root, rng)
    return root
