"""ASR suite: corpus WER overall and per accent."""

from __future__ import annotations

from pathlib import Path
from typing import Any

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


def run(data_root: Path, _cfg: dict[str, Any]) -> SuiteResult:
    path = manifest_path(data_root, ASR.name)
    if not path.exists():
        return no_data(ASR, data_root)
    if not systems.ASR:
        return SuiteResult(
            ASR.name,
            "not_configured",
            ASR.description,
            ASR.data_needed,
            notes=["No ASR provider configured (ASR_PROVIDER)."],
        )
    items = load_manifest(path, ("speaker_id", "audio", "reference_text"))
    result = SuiteResult(ASR.name, "measured", ASR.description, ASR.data_needed)
    for name, transcribe in systems.ASR.items():
        hyps = {it["id"]: transcribe(*io.read_wav_mono(resolve(it, "audio"))) for it in items}

        def wer_of(subset):
            r = metrics.corpus_wer([(it["reference_text"], hyps[it["id"]]) for it in subset])  # noqa: B023
            return {
                "wer": r.wer,
                "sub": r.substitutions,
                "del": r.deletions,
                "ins": r.insertions,
                "ref_words": r.reference_words,
            }

        subs = {f"accent={k}": wer_of(v) for k, v in group_by(items, "accent").items()}
        subs.update({f"gender={k}": wer_of(v) for k, v in group_by(items, "gender").items()})
        result.systems.append(SystemResult(name, wer_of(items), subs, n=len(items)))
    return result


ASR = Suite(
    "asr",
    "ASR corpus WER, overall and per accent",
    ">=2 h of interview-style answers from >=50 speakers with human reference transcripts; "
    "accent subgroup per speaker.",
    run,
)
