"""E9 performance evaluation: analyse the transcript and build the session report.

Every number is either measured from this session (with the transcript span or timestamp it
came from) or explicitly marked "not measured". Nothing is padded with synthetic values (the
prototype filled missing series with ``random.uniform``). Integrity notices are reported
separately from performance and do not change the coaching score.

The prototype's nine-factor score (``interview_core.legacy.grading``) is still computed and
included as an appendix, so the original E9 analysis remains reproducible on every report.
"""

from __future__ import annotations

from datetime import UTC, datetime
from statistics import mean
from typing import Any

from interview_core.legacy import grading as legacy_grading

DIMENSIONS = ("relevance", "structure", "depth", "communication", "technical_accuracy")


def _avg(values: list[float]) -> float | None:
    vals = [v for v in values if v is not None]
    return round(mean(vals), 3) if vals else None


def build_report(
    state: dict[str, Any],
    *,
    per_answer: list[dict[str, Any]] | None = None,
    integrity: dict[str, int] | None = None,
    mode: str = "coaching",
    code_results: list[dict[str, Any]] | None = None,
    generated_at: datetime | None = None,
) -> dict[str, Any]:
    """``state`` is ``InterviewState.to_dict()``; ``per_answer[i]`` holds optional observable
    signals for turn i: ``delivery`` (DeliveryMetrics dict), ``gaze`` (gaze summary dict),
    ``lipsync`` (AVSyncResult dict), ``voice`` (status/score)."""
    turns = state.get("turns", [])
    per_answer = per_answer or []
    answers = []
    for i, t in enumerate(turns):
        if t.get("answer") is None:
            continue
        ev = t.get("evaluation") or {}
        signals = per_answer[i] if i < len(per_answer) and per_answer[i] else {}
        answers.append(
            {
                "index": i,
                "question": t["question"]["question"],
                "category": t["question"]["category"],
                "difficulty": t["question"]["difficulty"],
                "follow_up": t.get("source") == "follow_up",
                "rationale": t["question"].get("rationale", ""),
                "answer": t["answer"],
                "scores": ev.get("scores"),
                "overall": ev.get("overall"),
                "label": ev.get("label", "experimental"),
                "method": ev.get("method"),
                "star": ev.get("star"),
                "evidence": ev.get("evidence", []),
                "strengths": ev.get("strengths", []),
                "improvements": ev.get("improvements", []),
                "delivery": signals.get("delivery") or {"measured": False},
                "gaze": signals.get("gaze") or {"measured": False},
                "lipsync": signals.get("lipsync") or {"measured": False},
                "voice": signals.get("voice") or {"measured": False},
            }
        )

    dims = {d: _avg([(a["scores"] or {}).get(d) for a in answers]) for d in DIMENSIONS}
    overall = _avg([a["overall"] for a in answers])
    labels = {a["label"] for a in answers}
    wpm = _avg([a["delivery"].get("words_per_minute") for a in answers])
    off = _avg([a["gaze"].get("off_screen_fraction") for a in answers])
    filler = _avg([a["delivery"].get("filler_per_100_words") for a in answers])

    observations = []
    for a in answers:
        g = a["gaze"]
        if g.get("off_screen_fraction") is not None and g.get("off_screen_fraction", 0) >= 0.3:
            observations.append(
                f"You looked away from the screen for {round(100 * g['off_screen_fraction'])}% "
                f"of answer {a['index'] + 1}."
            )
        d = a["delivery"]
        if d.get("longest_pause_s", 0) >= 3:
            observations.append(f"Answer {a['index'] + 1} had a {d['longest_pause_s']:.1f}s pause.")

    transcript = "\n".join(
        line
        for t in turns
        for line in (
            f"Interviewer: {t['question']['question']}",
            f"Candidate: {t['answer']}" if t.get("answer") is not None else "",
        )
        if line
    )
    legacy_score, legacy_breakdown, _ = legacy_grading.grade_interview_with_breakdown(
        transcript,
        state.get("resume_context", ""),
        job_role=state.get("role"),
        warning_count=sum((integrity or {}).values()) if mode == "proctored" else 0,
    )

    return {
        "version": 1,
        "generated_at": (generated_at or datetime.now(UTC)).isoformat(),
        "session_id": state.get("session_id"),
        "role": state.get("role"),
        "seniority": state.get("seniority"),
        "mode": mode,
        "summary": {
            "answers": len(answers),
            "overall": overall,
            "label": "calibrated" if labels == {"calibrated"} else "experimental",
            "dimensions": dims,
            "final_difficulty": state.get("difficulty"),
            "words_per_minute": wpm,
            "filler_per_100_words": filler,
            "off_screen_fraction": off,
        },
        "observations": observations,
        "answers": answers,
        "integrity": {
            "mode": mode,
            "events": integrity or {},
            "note": "Integrity notices are listed for your awareness and do not change your coaching score."
            if mode == "coaching"
            else "Proctored mode: notices are part of the session record.",
        },
        "code": code_results or [],
        "transcript": transcript,
        "appendix": {
            "prototype_nine_factor": {
                "score": legacy_score,
                "breakdown": legacy_breakdown,
                "label": "experimental (keyword heuristic, uncalibrated)",
            }
        },
    }
