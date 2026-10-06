"""Write docs/evidence/questions/README.md from the mock-interview JSON files.

Every number in the summary is computed from ``interview-*.json`` (written by
``eval/agent_mock_interviews.py``); nothing is typed by hand.

Usage:
  uv run python eval/summarize_mock_interviews.py docs/evidence/questions --source "run URL, commit"
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def row(path: Path) -> dict:
    d = json.loads(path.read_text(encoding="utf-8"))
    c, p = d["checks"], d["state"]["params"]
    ms = [a["ms"] for a in d.get("attempts", []) if a.get("ok")]
    return {
        "case": path.stem.removeprefix("interview-"),
        "role": f"{p['role']} ({p['seniority']})",
        "kind": f"{p['interview_type']}, {p['duration_minutes']} min, {p['language']}",
        "c": c,
        "ok_ms": ms,
        "calls": len(d.get("attempts", [])),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dir", type=Path)
    ap.add_argument("--source", default="", help="where the transcripts came from (run URL, commit)")
    a = ap.parse_args()
    rows = [row(p) for p in sorted(a.dir.glob("interview-*.json"))]
    if not rows:
        print("no interview-*.json files")
        return 1
    tot = lambda k: sum(r["c"].get(k, 0) for r in rows)  # noqa: E731
    passed = sum(r["c"]["passed"] for r in rows)
    ms = sorted(m for r in rows for m in r["ok_ms"])
    lines = [
        "# Mock interviews with the real LLM interviewer (P1 evidence)",
        "",
        "Each interview was run by `eval/agent_mock_interviews.py` against a real open-weights model. "
        "The candidate is simulated by the same model and given a profile (strong, average, weak, vague, mixed or improving). "
        "No question comes from a bank, and the checks below are computed by code, not by reading.",
        "",
        f"Source: {a.source or '-'}",
        "",
        "## Totals",
        "",
        f"- Interviews: {len(rows)}; passed every check: **{passed}/{len(rows)}**",
        f"- Questions asked: {tot('questions')}; follow-ups {tot('follow_ups')}, of which "
        f"{tot('follow_ups_referencing_answer')} quote and ask about the previous answer",
        f"- Overlap with the old question bank: {tot('bank_overlap')}; repeated questions: {tot('repeats')}",
        f"- Emergency (template) questions: {tot('emergency_questions')}; emergency blueprints: "
        f"{sum(bool(r['c'].get('emergency_blueprint')) for r in rows)}",
        f"- Questions written by the focused fresh-question call (D19): {tot('fresh_questions')}",
        f"- Difficulty moved against the candidate's performance: "
        f"{sum(not r['c']['difficulty_tracks_performance'] for r in rows)} interviews",
        f"- LLM calls: {sum(r['calls'] for r in rows)}; accepted-call latency p50 "
        f"{statistics.median(ms) / 1000:.1f} s, p95 {ms[int(0.95 * (len(ms) - 1))] / 1000:.1f} s "
        "(CPU-only CI runner; a GPU or hosted model is far faster)"
        if ms
        else "- LLM calls: -",
        "",
        "## Per interview",
        "",
        "| # | Role | Type | Qs | Follow-ups (grounded) | Repeats | Emergency | Fresh-call | Result |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        c = r["c"]
        lines.append(
            f"| [{r['case']}](interview-{r['case']}.md) | {r['role']} | {r['kind']} | {c['questions']} | "
            f"{c['follow_ups']} ({c['follow_ups_referencing_answer']}) | {c['repeats']} | "
            f"{c['emergency_questions']} | {c.get('fresh_questions', 0)} | {'PASS' if c['passed'] else 'FAIL'} |"
        )
    lines += [
        "",
        "A case fails if any emergency question or blueprint was used, a question repeats an earlier one "
        "or the bank, a follow-up does not quote and ask about the previous answer, or difficulty moved "
        "against the score.",
        "",
    ]
    (a.dir / "README.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"{passed}/{len(rows)} passed; wrote {a.dir / 'README.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
