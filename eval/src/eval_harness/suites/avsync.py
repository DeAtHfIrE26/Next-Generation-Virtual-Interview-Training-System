"""Lip-sync verification suite (E4): genuine vs mismatched audio-video."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from eval_harness import io, metrics, systems
from eval_harness.base import Suite, SuiteResult, SystemResult, load_manifest, manifest_path, no_data, resolve


def run(data_root: Path, cfg: dict[str, Any]) -> SuiteResult:
    path = manifest_path(data_root, LIPSYNC.name)
    if not path.exists():
        return no_data(LIPSYNC, data_root)
    items = load_manifest(path, ("subject_id", "audio", "mouth_series", "label"))
    result = SuiteResult(LIPSYNC.name, "measured", LIPSYNC.description, LIPSYNC.data_needed)
    loaded = [
        (it, *io.read_wav_mono(resolve(it, "audio")), io.read_json(resolve(it, "mouth_series")))
        for it in items
    ]
    kinds = sorted({it.get("mismatch_kind", "unspecified") for it in items if it["label"] == "mismatch"})
    for name, scorer in systems.LIPSYNC.items():
        scored = [(it, scorer(wav, sr, series)) for it, wav, sr, series in loaded]
        gen = [s for it, s in scored if it["label"] == "genuine"]
        bad = [s for it, s in scored if it["label"] == "mismatch"]
        m: dict[str, Any] = {"genuine": len(gen), "mismatch": len(bad)}
        if gen and bad:
            m["auc"] = metrics.roc_auc(gen, bad)
            m["eer"], m["eer_threshold"] = metrics.equal_error_rate(gen, bad)
            thr = cfg.get("lipsync_threshold", {}).get(name)
            if thr is not None:
                m["false_alarm_rate"] = sum(s < thr for s in gen) / len(gen)
                m["miss_rate"] = sum(s >= thr for s in bad) / len(bad)
            sub = {}
            for k in kinds:
                kb = [
                    s
                    for it, s in scored
                    if it["label"] == "mismatch" and it.get("mismatch_kind", "unspecified") == k
                ]
                sub[f"mismatch_kind={k}"] = {"auc": metrics.roc_auc(gen, kb), "n": len(kb)}
        else:
            sub = {}
            m["error"] = "need genuine and mismatch clips"
        result.systems.append(SystemResult(name, m, sub, n=len(scored)))
    return result


LIPSYNC = Suite(
    "lipsync",
    "E4 lip-sync verification: AUC / EER separating genuine from mismatched clips",
    ">=100 subjects x 5 genuine 10-20 s clips (16 kHz WAV plus per-frame face landmarks JSON); "
    "constructed negatives (other-speaker audio, +/-100-500 ms offsets, playback while silent) "
    "labelled with mismatch_kind.",
    run,
)
