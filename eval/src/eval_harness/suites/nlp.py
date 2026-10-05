"""NLP suites (E6): answer-score agreement, question relevance, LLM schema validity."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from eval_harness import metrics, systems
from eval_harness.base import (
    Suite,
    SuiteResult,
    SystemResult,
    group_by,
    load_manifest,
    manifest_path,
    no_data,
)

# Calibration gate used by the product: below these the score is labelled "experimental".
CALIBRATION_GATE = {"spearman": 0.6, "qwk": 0.6}


def _to_scale(x: float, lo: int = 1, hi: int = 5) -> int:
    return int(np.clip(round(lo + x * (hi - lo)), lo, hi))


def run_answer_scoring(data_root: Path, _cfg: dict[str, Any]) -> SuiteResult:
    path = manifest_path(data_root, ANSWER.name)
    if not path.exists():
        return no_data(ANSWER, data_root)
    items = load_manifest(path, ("question", "answer", "human_scores"))
    result = SuiteResult(ANSWER.name, "measured", ANSWER.description, ANSWER.data_needed)
    human = [float(np.mean(it["human_scores"])) for it in items]  # 1-5 rubric
    rater_cols = list(
        zip(*[it["human_scores"] for it in items if len(it["human_scores"]) >= 2], strict=False)
    )
    if len(rater_cols) >= 2:
        result.notes.append(
            f"Inter-rater QWK (raters 1 vs 2): {metrics.cohen_kappa(rater_cols[0], rater_cols[1], weights='quadratic'):.3f}"
        )

    def agreement(sys_scores, subset_idx):
        h = [human[i] for i in subset_idx]
        s = [sys_scores[i] for i in subset_idx]
        if len(h) < 3:
            return {"n": len(h), "note": "too few items"}
        rho = metrics.spearman(s, h)
        qwk = metrics.cohen_kappa([_to_scale(x) for x in s], [round(x) for x in h], weights="quadratic")
        return {
            "n": len(h),
            "spearman": rho,
            "qwk": qwk,
            "calibrated": rho >= CALIBRATION_GATE["spearman"] and qwk >= CALIBRATION_GATE["qwk"],
        }

    for name, scorer in systems.ANSWER.items():
        scores = [
            scorer(it["question"], it["answer"], it.get("role", ""), it.get("context", "")) for it in items
        ]
        idx_all = list(range(len(items)))
        subs = {}
        for key in ("accent", "gender", "seniority"):
            for val, members in group_by(items, key).items():
                ids = {id(m) for m in members}
                subs[f"{key}={val}"] = agreement(scores, [i for i, it in enumerate(items) if id(it) in ids])
        result.systems.append(SystemResult(name, agreement(scores, idx_all), subs, n=len(items)))
    return result


def run_question_relevance(data_root: Path, _cfg: dict[str, Any]) -> SuiteResult:
    path = manifest_path(data_root, QUESTIONS.name)
    if not path.exists():
        return no_data(QUESTIONS, data_root)
    items = load_manifest(path, ("generator", "question", "ratings"))
    result = SuiteResult(QUESTIONS.name, "measured", QUESTIONS.description, QUESTIONS.data_needed)
    by_gen: dict[str, list[dict[str, Any]]] = {}
    for it in items:
        by_gen.setdefault(it["generator"], []).append(it)
    for gen, its in by_gen.items():
        ratings = [it["ratings"] for it in its]
        m: dict[str, Any] = {
            "mean_relevance_1to5": float(np.mean([np.mean(r) for r in ratings])),
            "share_rated_relevant(>=4)": float(np.mean([np.mean(r) >= 4 for r in ratings])),
        }
        pairs = [r for r in ratings if len(r) >= 2]
        if len(pairs) >= 2:
            m["inter_rater_qwk"] = metrics.cohen_kappa(
                [r[0] for r in pairs], [r[1] for r in pairs], weights="quadratic"
            )
        result.systems.append(SystemResult(gen, m, n=len(its)))
    return result


def run_schema_validity(data_root: Path, _cfg: dict[str, Any]) -> SuiteResult:
    """Reads LLM call logs exported by the API (``eval/data/llm_schema_validity/manifest.jsonl``)."""
    path = manifest_path(data_root, SCHEMA.name)
    if not path.exists():
        return no_data(SCHEMA, data_root)
    items = load_manifest(path, ("provider", "task", "raw_valid", "delivered_valid"))
    result = SuiteResult(SCHEMA.name, "measured", SCHEMA.description, SCHEMA.data_needed)
    for prov in sorted({it["provider"] for it in items}):
        its = [it for it in items if it["provider"] == prov]
        m = {
            "calls": len(its),
            "raw_schema_valid_rate": float(np.mean([bool(it["raw_valid"]) for it in its])),
            "delivered_valid_rate": float(np.mean([bool(it["delivered_valid"]) for it in its])),
            "fallback_rate": float(np.mean([bool(it.get("used_fallback")) for it in its])),
        }
        result.systems.append(SystemResult(prov, m, n=len(its)))
    return result


ANSWER = Suite(
    "answer_scoring",
    "E6/E9 answer-score agreement with human raters (Spearman, QWK), by subgroup",
    ">=300 answers from >=60 consenting candidates, each rated 1-5 on the published rubric by "
    ">=3 trained raters.",
    run_answer_scoring,
)
QUESTIONS = Suite(
    "question_relevance",
    "E6 question relevance as rated by experts, per generator version",
    "200 generated questions from >=40 consented, de-identified resume/JD pairs, each rated 1-5 "
    "by 3 experts.",
    run_question_relevance,
)
SCHEMA = Suite(
    "llm_schema_validity",
    "LLM structured-output validity before and after validation/fallback",
    "Exported LLM call log from staging or E2E runs (consent_id = 'system' for synthetic sessions).",
    run_schema_validity,
)
