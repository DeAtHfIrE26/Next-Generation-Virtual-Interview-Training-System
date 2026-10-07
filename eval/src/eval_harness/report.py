"""Render suite results as docs/EVAL_REPORT.md and a JSON artifact."""

from __future__ import annotations

import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from eval_harness.base import SuiteResult

STATUS_TEXT = {
    "measured": "Measured on real, consented data",
    "no_data": "NOT MEASURED: no evaluation data yet",
    "not_configured": "NOT MEASURED: system under test not configured",
    "smoke": "Smoke run on synthetic data (pipeline check only, not accuracy)",
    "error": "ERROR",
}


def _fmt(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:.4f}"
    if isinstance(v, dict):
        return ", ".join(f"{k}={_fmt(x)}" for k, x in v.items())
    return str(v)


def _git_rev() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def render_markdown(results: list[SuiteResult], *, smoke: list[SuiteResult] | None = None) -> str:
    lines = [
        "# Evaluation Report",
        "",
        f"Generated {datetime.now(UTC).strftime('%Y-%m-%d %H:%M UTC')} at commit `{_git_rev()}` by "
        "`python -m eval_harness run`.",
        "",
        "**Rule:** a number appears here only if it was measured on a real, consented evaluation set "
        "described by a manifest. Synthetic smoke runs check that pipelines execute and never report accuracy. "
        'Systems named `legacy_*` are the prototype algorithms (the "before" numbers); the others are the '
        'upgraded implementations ("after").',
        "",
        "## Summary",
        "",
        "| Suite | Status | Systems |",
        "|---|---|---|",
    ]
    for r in results:
        lines.append(
            f"| {r.suite} | {STATUS_TEXT[r.status]} | {', '.join(s.system for s in r.systems) or '-'} |"
        )
    for r in results:
        lines += [
            "",
            f"## {r.suite}",
            "",
            f"{r.description}.",
            "",
            f"**Status:** {STATUS_TEXT[r.status]}",
            "",
        ]
        if r.status != "measured":
            lines.append(f"**Data needed:** {r.data_needed}")
        for note in r.notes:
            lines.append(f"- {note}")
        for s in r.systems:
            lines += ["", f"### {s.system} (n={s.n})", "", "| Metric | Value |", "|---|---|"]
            lines += [f"| {k} | {_fmt(v)} |" for k, v in s.metrics.items()]
            if s.subgroups:
                lines += ["", "| Subgroup | Metrics |", "|---|---|"]
                lines += [f"| {k} | {_fmt(v)} |" for k, v in s.subgroups.items()]
            for note in s.notes:
                lines.append(f"- {note}")
    if smoke:
        lines += [
            "",
            "## Smoke runs (synthetic data, pipeline check only)",
            "",
            "| Suite | Ran | Systems exercised | Items |",
            "|---|---|---|---|",
        ]
        for r in smoke:
            ok = "yes" if r.status == "smoke" else f"no ({r.status})"
            lines.append(
                f"| {r.suite} | {ok} | {', '.join(s.system for s in r.systems)} | "
                f"{max((s.n for s in r.systems), default=0)} |"
            )
    return "\n".join(lines) + "\n"


def write(
    results: list[SuiteResult],
    out_dir: Path,
    report_path: Path | None,
    smoke: list[SuiteResult] | None = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {"results": [r.to_dict() for r in results], "smoke": [r.to_dict() for r in smoke or []]}
    (out_dir / "results.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(render_markdown(results, smoke=smoke), encoding="utf-8")
