"""E9 performance evaluation: analyse the interview and build the session report.

Every number is either measured from this session (with the transcript span or timestamp it
came from) or explicitly marked "not measured". Nothing is padded with synthetic values (the
prototype filled missing series with ``random.uniform``). Integrity notices are reported
separately from performance and do not change the coaching score.

Inputs: the interviewer agent's state (``AgentState.to_dict()``) where each answered turn carries
the evidence-verified rubric evaluation (``Turn.evaluation``) and the agent's live 1-5 read
(``Turn.score``), plus per-answer observable signals.

The prototype's nine-factor score (``interview_core.legacy.grading``) is still computed and
included as an appendix, so the original E9 analysis remains reproducible on every report.
"""

from __future__ import annotations

from datetime import UTC, datetime
from statistics import mean
from typing import Any

from interview_core.legacy import grading as legacy_grading

DIMENSIONS = ("relevance", "structure", "depth", "communication", "technical_accuracy")
VERSION = 2


def _avg(values: list[float | None]) -> float | None:
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
    params = state.get("params", {})
    blueprint = state.get("blueprint") or {}
    comp_names = {c["id"]: c["name"] for c in blueprint.get("competencies", [])}
    turns = state.get("turns", [])
    per_answer = per_answer or []

    answers = []
    for t in turns:
        if t.get("answer") is None:
            continue
        i = t["index"]
        ev = t.get("evaluation") or {}
        signals = per_answer[i] if i < len(per_answer) and per_answer[i] else {}
        answers.append(
            {
                "index": i,
                "question": t["say"],
                "action": t["action"],
                "competency": comp_names.get(t["competency"], t["competency"]),
                "competency_id": t["competency"],
                "difficulty": t["difficulty"],
                "emergency_question": bool(t.get("emergency")),
                "anchor_quote": t.get("anchor_quote", ""),
                "answer": t["answer"],
                "answer_seconds": t.get("answer_seconds", 0.0),
                "interviewer_read": t.get("score"),
                "scores": ev.get("scores"),
                "overall": ev.get("overall"),
                "label": ev.get("label", "experimental"),
                "method": ev.get("method", "not evaluated"),
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

    # Per-competency (skill) breakdown from the blueprint
    skills = []
    for c in blueprint.get("competencies", []):
        mine = [a for a in answers if a["competency_id"] == c["id"]]
        skills.append(
            {
                "id": c["id"],
                "name": c["name"],
                "why": c.get("why", ""),
                "questions": len(mine),
                "overall": _avg([a["overall"] for a in mine]),
                "interviewer_read": _avg([a["interviewer_read"] for a in mine]),
                "dimensions": {d: _avg([(a["scores"] or {}).get(d) for a in mine]) for d in DIMENSIONS},
                "covered": bool(mine),
            }
        )

    # Highlighted moments: strongest and weakest answers, each with the candidate's own words
    rated = [a for a in answers if a["overall"] is not None]
    moments = []
    if rated:
        best = max(rated, key=lambda a: a["overall"])
        worst = min(rated, key=lambda a: a["overall"])
        for kind, a in (("strongest", best), ("needs_work", worst)):
            if kind == "needs_work" and a is best and len(rated) > 1:
                continue
            quote = (a["evidence"] or [{}])[0].get("quote") if a["evidence"] else None
            moments.append(
                {
                    "kind": kind,
                    "index": a["index"],
                    "question": a["question"],
                    "quote": quote,
                    "overall": a["overall"],
                    "why": (a["strengths"] if kind == "strongest" else a["improvements"])[:2],
                }
            )

    # Concrete tips: the most frequent improvement themes, with the answers they came from
    tips: list[dict[str, Any]] = []
    for a in sorted(rated, key=lambda a: a["overall"])[:4]:
        for imp in a["improvements"][:1]:
            tips.append({"tip": imp, "from_answer": a["index"]})

    wpm = _avg([a["delivery"].get("words_per_minute") for a in answers])
    off = _avg([a["gaze"].get("off_screen_fraction") for a in answers])
    filler = _avg([a["delivery"].get("filler_per_100_words") for a in answers])
    observations = []
    for a in answers:
        g = a["gaze"]
        if g.get("off_screen_fraction") is not None and g.get("off_screen_fraction", 0) >= 0.3:
            observations.append(
                f"You looked away from the screen for {round(100 * g['off_screen_fraction'])}% of answer {a['index'] + 1}."
            )
        d = a["delivery"]
        if d.get("longest_pause_s", 0) >= 3:
            observations.append(f"Answer {a['index'] + 1} had a {d['longest_pause_s']:.1f}s pause.")

    transcript_lines = []
    for t in turns:
        transcript_lines.append({"speaker": "interviewer", "text": t["say"], "index": t["index"]})
        if t.get("answer") is not None:
            transcript_lines.append({"speaker": "candidate", "text": t["answer"], "index": t["index"]})
    transcript_text = "\n".join(
        f"{'Interviewer' if ln['speaker'] == 'interviewer' else 'Candidate'}: {ln['text']}"
        for ln in transcript_lines
    )
    legacy_score, legacy_breakdown, _ = legacy_grading.grade_interview_with_breakdown(
        transcript_text,
        params.get("resume_context", ""),
        job_role=params.get("role"),
        warning_count=sum((integrity or {}).values()) if mode == "proctored" else 0,
    )
    emergency = sum(1 for t in turns if t.get("emergency"))

    return {
        "version": VERSION,
        "generated_at": (generated_at or datetime.now(UTC)).isoformat(),
        "session_id": state.get("session_id"),
        "role": params.get("role"),
        "seniority": params.get("seniority"),
        "company": params.get("company") or None,
        "interview_type": params.get("interview_type"),
        "round": params.get("round"),
        "mode": mode,
        "summary": {
            "answers": len(answers),
            "overall": overall,
            "label": "calibrated" if labels == {"calibrated"} else "experimental",
            "dimensions": dims,
            "difficulty_trajectory": [t["difficulty"] for t in turns],
            "final_difficulty": state.get("difficulty"),
            "words_per_minute": wpm,
            "filler_per_100_words": filler,
            "off_screen_fraction": off,
            "emergency_questions": emergency,
            "duration_minutes": params.get("duration_minutes"),
        },
        "blueprint": {"summary": blueprint.get("summary", ""), "emergency": bool(blueprint.get("emergency"))},
        "skills": skills,
        "moments": moments,
        "tips": tips,
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
        "transcript": transcript_lines,
        "appendix": {
            "prototype_nine_factor": {
                "score": legacy_score,
                "breakdown": legacy_breakdown,
                "label": "experimental (keyword heuristic, uncalibrated)",
            }
        },
    }
