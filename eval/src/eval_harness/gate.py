"""Regression gate: compare a results.json against a stored baseline.

A metric regresses when it moves in the bad direction by more than its tolerance. Only
metrics measured in both runs are compared; smoke results are ignored.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# metric -> (direction, absolute tolerance or relative tolerance with "rel")
TOLERANCES: dict[str, tuple[str, float, str]] = {
    "eer": ("lower", 0.005, "abs"),
    "far": ("lower", 0.002, "abs"),
    "frr": ("lower", 0.01, "abs"),
    "wer": ("lower", 0.01, "abs"),
    "auc": ("higher", 0.01, "abs"),
    "spearman": ("higher", 0.02, "abs"),
    "qwk": ("higher", 0.02, "abs"),
    "p95_ms": ("lower", 0.10, "rel"),
    "raw_schema_valid_rate": ("higher", 0.01, "abs"),
    "delivered_valid_rate": ("higher", 0.0, "abs"),
    "offscreen_f1": ("higher", 0.02, "abs"),
}


def _index(payload: dict[str, Any]) -> dict[tuple[str, str, str], float]:
    out = {}
    for r in payload.get("results", []):
        if r["status"] != "measured":
            continue
        for s in r["systems"]:
            for k, v in s["metrics"].items():
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    out[(r["suite"], s["system"], k)] = float(v)
    return out


def compare(current: dict[str, Any], baseline: dict[str, Any]) -> list[str]:
    cur, base = _index(current), _index(baseline)
    failures = []
    for key, b in base.items():
        metric = key[2]
        if metric not in TOLERANCES or key not in cur:
            continue
        direction, tol, kind = TOLERANCES[metric]
        allowed = tol * abs(b) if kind == "rel" else tol
        c = cur[key]
        worse = (c - b) if direction == "lower" else (b - c)
        if worse > allowed:
            failures.append(f"{key[0]}/{key[1]}/{metric}: {b:.4f} -> {c:.4f} (tolerance {allowed:.4f})")
    return failures


def run_gate(current_path: Path, baseline_path: Path) -> list[str]:
    if not baseline_path.exists():
        return []
    return compare(json.loads(current_path.read_text()), json.loads(baseline_path.read_text()))
