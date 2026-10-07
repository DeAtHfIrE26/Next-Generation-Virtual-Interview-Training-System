"""Systems exposed to the evaluation harness (eval_harness.upgraded)."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from interview_core.nlp import evaluator, heuristics, providers
from interview_core.nlp.structured import StructuredLLM
from interview_core.speech import asr


def _question(q: str) -> dict:
    return {"question": q, "category": "behavioral", "difficulty": 3, "expected_points": []}


def eval_answer_scorers() -> dict[str, Callable[[str, str, str, str], float]]:
    out: dict[str, Callable[[str, str, str, str], float]] = {
        "heuristic_v2": lambda q, a, _r, _c: evaluator.overall_score(
            heuristics.heuristic_evaluation(_question(q), a)
        ),
    }
    prov = providers.from_env()
    if prov is not None:
        llm = StructuredLLM(prov)
        out[f"llm_rubric:{prov.model}"] = lambda q, a, r, _c: (
            evaluator.evaluate(llm, _question(q), a, role=r or "candidate", seniority="mid").overall
        )
    return out


def eval_asr_systems() -> dict[str, Callable[[np.ndarray, int], str]]:
    """The live interview's streaming STT (local sherpa-onnx by default) plus any batch ASR adapter."""
    from interview_core.realtime import assets, stt

    out: dict[str, Callable[[np.ndarray, int], str]] = {}
    if assets.is_present("stt-streaming"):
        live = stt.sherpa_provider()
        out["sherpa:nemotron-streaming+parakeet-final"] = lambda audio, sr: (
            stt.transcribe(live, audio, sr).text
        )
    p = asr.from_env()
    if p is not None:
        out[p.name] = lambda audio, sr: p.transcribe(audio, sr).text
    return out
