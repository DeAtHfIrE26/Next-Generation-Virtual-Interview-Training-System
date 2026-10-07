"""Face verification, liveness and speaker verification suites (E2, E3)."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from eval_harness import io, metrics, systems
from eval_harness.base import (
    Suite,
    SuiteResult,
    SystemResult,
    group_by,
    load_manifest,
    manifest_path,
    no_data,
    resolve,
)

SUBGROUP_KEYS = ("gender", "age_band", "skin_tone_monk", "glasses", "device", "lighting", "accent")


def _pairs_by_subject(items: list[dict[str, Any]]):
    enrol: dict[str, list[dict[str, Any]]] = defaultdict(list)
    probes: list[dict[str, Any]] = []
    for it in items:
        (enrol[it["subject_id"]] if it["role"] == "enroll" else probes).append(it)
    return enrol, probes


def _verification_metrics(scored: list[tuple[dict[str, Any], float, bool]], threshold: float | None):
    gen = [s for _, s, same in scored if same]
    imp = [s for _, s, same in scored if not same]
    if not gen or not imp:
        return {"error": "need genuine and impostor pairs"}
    rep = metrics.verification_report(gen, imp, threshold)
    out = {
        "eer": rep.eer,
        "eer_threshold": rep.eer_threshold,
        "genuine_pairs": rep.genuine,
        "impostor_pairs": rep.impostor,
        "threshold_at_far_0.1pct": metrics.threshold_for_far(imp, 0.001),
    }
    if threshold is not None:
        out.update(far=rep.far_at_threshold, frr=rep.frr_at_threshold, threshold=threshold)
    return out


def _subgroups(scored, threshold):
    out: dict[str, dict[str, Any]] = {}
    for key in SUBGROUP_KEYS:
        groups = group_by([p for p, _, _ in scored], key)
        for val, members in groups.items():
            ids = {id(m) for m in members}
            sub = [t for t in scored if id(t[0]) in ids]
            out[f"{key}={val}"] = _verification_metrics(sub, threshold)
    return out


def run_face(data_root: Path, cfg: dict[str, Any]) -> SuiteResult:
    path = manifest_path(data_root, FACE.name)
    if not path.exists():
        return no_data(FACE, data_root)
    items = load_manifest(path, ("subject_id", "role", "image"))
    enrol, probes = _pairs_by_subject(items)
    images = {it["id"]: io.read_gray(resolve(it, "image")) for it in items}
    result = SuiteResult(FACE.name, "measured", FACE.description, FACE.data_needed)
    for name, scorer in systems.FACE.items():
        scored = []
        for subject, refs in enrol.items():
            ref_imgs = [images[r["id"]] for r in refs]
            for p in probes:
                scored.append((p, scorer(ref_imgs, images[p["id"]]), p["subject_id"] == subject))
        thr = cfg.get("face_threshold", {}).get(name)
        result.systems.append(
            SystemResult(name, _verification_metrics(scored, thr), _subgroups(scored, thr), n=len(scored))
        )
    return result


def run_speaker(data_root: Path, cfg: dict[str, Any]) -> SuiteResult:
    path = manifest_path(data_root, SPEAKER.name)
    if not path.exists():
        return no_data(SPEAKER, data_root)
    if not systems.SPEAKER:
        return SuiteResult(
            SPEAKER.name,
            "not_configured",
            SPEAKER.description,
            SPEAKER.data_needed,
            notes=["No speaker embedder configured (SPEAKER_EMBEDDER)."],
        )
    items = load_manifest(path, ("subject_id", "role", "audio"))
    enrol, probes = _pairs_by_subject(items)
    audio = {it["id"]: io.read_wav_mono(resolve(it, "audio")) for it in items}
    result = SuiteResult(SPEAKER.name, "measured", SPEAKER.description, SPEAKER.data_needed)
    for name, scorer in systems.SPEAKER.items():
        scored, spoof_scored = [], []
        for subject, refs in enrol.items():
            ref_audio = [audio[r["id"]][0] for r in refs]
            for p in probes:
                wav, sr = audio[p["id"]]
                s = scorer(ref_audio, wav, sr)
                if p.get("kind", "genuine") == "genuine":
                    scored.append((p, s, p["subject_id"] == subject))
                elif p["subject_id"] == subject:
                    spoof_scored.append(s)  # replay/clone of the enrolled speaker
        thr = cfg.get("voice_threshold", {}).get(name)
        m = _verification_metrics(scored, thr)
        if spoof_scored and thr is not None:
            m["spoof_acceptance_rate"] = float(np.mean(np.asarray(spoof_scored) >= thr))
        result.systems.append(SystemResult(name, m, _subgroups(scored, thr), n=len(scored)))
    return result


def run_liveness(data_root: Path, cfg: dict[str, Any]) -> SuiteResult:
    path = manifest_path(data_root, LIVENESS.name)
    if not path.exists():
        return no_data(LIVENESS, data_root)
    if not systems.LIVENESS:
        return SuiteResult(LIVENESS.name, "not_configured", LIVENESS.description, LIVENESS.data_needed)
    items = load_manifest(path, ("subject_id", "kind", "series"))
    result = SuiteResult(LIVENESS.name, "measured", LIVENESS.description, LIVENESS.data_needed)
    for name, scorer in systems.LIVENESS.items():
        scores = [(it, scorer(io.read_json(resolve(it, "series")))) for it in items]
        bona = [s for it, s in scores if it["kind"] == "bona_fide"]
        thr = cfg.get("liveness_threshold", {}).get(name, 0.5)
        m: dict[str, Any] = {"threshold": thr, "bona_fide": len(bona)}
        for kind in sorted({it["kind"] for it in items} - {"bona_fide"}):
            attacks = [s for it, s in scores if it["kind"] == kind]
            apcer, bpcer = metrics.apcer_bpcer(attacks, bona, thr)
            m[f"apcer[{kind}]"] = apcer
            m["bpcer"] = bpcer
        result.systems.append(SystemResult(name, m, n=len(items)))
    return result


FACE = Suite(
    "face_verification",
    "E2 face verification: FAR / FRR / EER, by subgroup",
    ">=100 subjects; per subject 1 enrolment session (>=15 aligned face crops) and >=10 probe "
    "crops across 2 lighting conditions and >=2 devices; optional self-reported subgroups.",
    run_face,
)
SPEAKER = Suite(
    "speaker_verification",
    "E3 speaker verification EER and spoof acceptance, by accent",
    ">=100 speakers (mostly Indian-English), 3 sessions, >=2 devices; replay and TTS-clone "
    "probes made only from each consenting speaker's own voice.",
    run_speaker,
)
LIVENESS = Suite(
    "liveness",
    "E2 liveness: APCER per attack type and BPCER (ISO/IEC 30107-3 style)",
    ">=50 subjects x {bona fide, printed photo, phone-screen replay, laptop video replay}; "
    "landmark series JSON per attempt.",
    run_liveness,
)
