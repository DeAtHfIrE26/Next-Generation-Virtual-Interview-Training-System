"""Eye-tracking suite (E5): frame-level on-screen / off-screen agreement with annotators."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from eval_harness import io, metrics, systems
from eval_harness.base import Suite, SuiteResult, SystemResult, load_manifest, manifest_path, no_data, resolve


def run(data_root: Path, _cfg: dict[str, Any]) -> SuiteResult:
    path = manifest_path(data_root, GAZE.name)
    if not path.exists():
        return no_data(GAZE, data_root)
    items = load_manifest(path, ("subject_id", "frames"))
    result = SuiteResult(GAZE.name, "measured", GAZE.description, GAZE.data_needed)
    frames = []
    for it in items:
        clip = io.read_json(resolve(it, "frames"))
        for fr in clip["frames"]:
            if fr.get("label_on_screen") is None:
                continue
            frames.append((it, fr, clip["width"], clip["height"]))
    for name, predict in systems.GAZE.items():
        truth, pred, no_face = [], [], 0
        for _it, fr, w, h in frames:
            lm = np.asarray(fr["landmarks"]) if fr.get("landmarks") else None
            p = predict(lm, w, h)
            if p is None:
                no_face += 1
                continue
            truth.append(bool(fr["label_on_screen"]))
            pred.append(bool(p))
        m: dict[str, Any] = {"frames": len(truth), "no_face_frames": no_face}
        if truth:
            tp = sum((not t) and (not p) for t, p in zip(truth, pred, strict=True))  # positive = off-screen
            fp = sum(t and (not p) for t, p in zip(truth, pred, strict=True))
            fn = sum((not t) and p for t, p in zip(truth, pred, strict=True))
            m["offscreen_precision"], m["offscreen_recall"], m["offscreen_f1"] = metrics.precision_recall(
                tp, fp, fn
            )
            m["cohen_kappa"] = metrics.cohen_kappa(truth, pred) if len(set(truth) | set(pred)) > 1 else None
            m["accuracy"] = float(np.mean(np.asarray(truth) == np.asarray(pred)))
        result.systems.append(SystemResult(name, m, n=len(truth)))
    return result


GAZE = Suite(
    "gaze",
    "E5 gaze: off-screen precision / recall and Cohen's kappa vs human annotation",
    ">=30 subjects x 2 min interview-style video, per-frame landmarks plus on/off-screen labels "
    "from 2 annotators (disagreements removed or adjudicated).",
    run,
)
