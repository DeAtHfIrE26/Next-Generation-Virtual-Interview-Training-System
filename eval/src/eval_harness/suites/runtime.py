"""Latency, device-detection and avatar lip-sync (LSE) suites."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from eval_harness import metrics
from eval_harness.base import Suite, SuiteResult, SystemResult, load_manifest, manifest_path, no_data

TARGETS_MS = {"question_to_first_avatar_frame": 1500.0}


def run_latency(data_root: Path, _cfg: dict[str, Any]) -> SuiteResult:
    path = manifest_path(data_root, LATENCY.name)
    if not path.exists():
        return no_data(LATENCY, data_root)
    items = load_manifest(path, ("metric", "ms"))
    result = SuiteResult(LATENCY.name, "measured", LATENCY.description, LATENCY.data_needed)
    for name in sorted({it["metric"] for it in items}):
        summary: dict[str, Any] = metrics.latency_summary([it["ms"] for it in items if it["metric"] == name])
        if name in TARGETS_MS:
            summary["target_p95_ms"] = TARGETS_MS[name]
            summary["meets_target"] = summary["p95_ms"] < TARGETS_MS[name]
        result.systems.append(SystemResult(name, summary, n=summary["n"]))
    return result


def run_detection(data_root: Path, _cfg: dict[str, Any]) -> SuiteResult:
    """Ground-truth labels vs predictions exported from the detector under test."""
    path = manifest_path(data_root, DETECTION.name)
    if not path.exists():
        return no_data(DETECTION, data_root)
    items = load_manifest(path, ("detector", "event", "truth", "predicted"))
    result = SuiteResult(DETECTION.name, "measured", DETECTION.description, DETECTION.data_needed)
    for det in sorted({it["detector"] for it in items}):
        m: dict[str, Any] = {}
        for ev in sorted({it["event"] for it in items if it["detector"] == det}):
            its = [it for it in items if it["detector"] == det and it["event"] == ev]
            tp = sum(bool(it["truth"]) and bool(it["predicted"]) for it in its)
            fp = sum((not it["truth"]) and bool(it["predicted"]) for it in its)
            fn = sum(bool(it["truth"]) and not it["predicted"] for it in its)
            p, r, f1 = metrics.precision_recall(tp, fp, fn)
            m[ev] = {"precision": p, "recall": r, "f1": f1, "frames": len(its)}
        result.systems.append(SystemResult(det, m, n=sum(1 for it in items if it["detector"] == det)))
    return result


def run_avatar_sync(data_root: Path, _cfg: dict[str, Any]) -> SuiteResult:
    """Summarises LSE-D / LSE-C computed by the reference SyncNet evaluation script."""
    path = manifest_path(data_root, AVATAR.name)
    if not path.exists():
        return no_data(AVATAR, data_root)
    items = load_manifest(path, ("avatar_path", "lse_d", "lse_c"))
    result = SuiteResult(AVATAR.name, "measured", AVATAR.description, AVATAR.data_needed)
    for av in sorted({it["avatar_path"] for it in items}):
        its = [it for it in items if it["avatar_path"] == av]
        result.systems.append(
            SystemResult(
                av,
                {
                    "lse_d_mean": float(np.mean([it["lse_d"] for it in its])),
                    "lse_c_mean": float(np.mean([it["lse_c"] for it in its])),
                },
                n=len(its),
            )
        )
    return result


LATENCY = Suite(
    "latency",
    "End-to-end latency p50 / p95 per stage (target: first avatar frame p95 < 1.5 s)",
    "Latency events exported from instrumented sessions (API /metrics/latency export).",
    run_latency,
)
DETECTION = Suite(
    "device_detection",
    "E7 phone / second-person detection precision and recall",
    ">=500 annotated frames from consented sessions; predictions exported from the detector.",
    run_detection,
)
AVATAR = Suite(
    "avatar_sync",
    "Avatar lip-sync quality (LSE-D lower is better, LSE-C higher is better)",
    "100 generated clips per avatar path scored with the SyncNet evaluation script (weights "
    "licence must be checked before use).",
    run_avatar_sync,
)
