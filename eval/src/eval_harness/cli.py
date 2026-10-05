"""Command line: ``python -m eval_harness {run,gate,validate}``."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

from eval_harness import gate, report, synthetic, systems
from eval_harness.base import DEFAULT_DATA_ROOT, ManifestError, SuiteResult, load_manifest, manifest_path
from eval_harness.suites import ALL, BY_NAME

REPO_ROOT = Path(__file__).resolve().parents[3]


def _run_suites(names: list[str], data_root: Path, cfg: dict) -> list[SuiteResult]:
    out = []
    for name in names:
        suite = BY_NAME[name]
        try:
            out.append(suite.run(data_root, cfg))
        except (ManifestError, ValueError, OSError) as e:
            out.append(SuiteResult(suite.name, "error", suite.description, suite.data_needed, notes=[str(e)]))
    return out


def run_smoke(cfg: dict) -> list[SuiteResult]:
    with tempfile.TemporaryDirectory() as tmp:
        root = synthetic.generate(Path(tmp))
        names = [s.name for s in ALL if manifest_path(root, s.name).exists()]
        results = _run_suites(names, root, cfg)
    for r in results:
        if r.status == "measured":
            r.status = "smoke"
            for s in r.systems:
                s.metrics = {}  # never surface synthetic numbers
                s.subgroups = {}
    return results


def cmd_run(args: argparse.Namespace) -> int:
    cfg = json.loads(Path(args.config).read_text()) if args.config else {}
    notes = systems.register_optional()
    smoke = run_smoke(cfg) if args.suite in ("all", "smoke") else []
    results = (
        []
        if args.suite == "smoke"
        else _run_suites(
            [s.name for s in ALL] if args.suite == "all" else [args.suite], Path(args.data_root), cfg
        )
    )
    if results and notes:
        results[0].notes.extend(notes)
    report_path = Path(args.report) if args.report else None
    if args.suite == "smoke" and args.report is None:
        report_path = None
    report.write(results, Path(args.out), report_path, smoke=smoke)
    failed = [r.suite for r in smoke if r.status != "smoke"] + [
        r.suite for r in results if r.status == "error"
    ]
    for r in smoke + results:
        print(f"{r.suite:22s} {r.status}")
    if failed:
        print(f"FAILED: {failed}", file=sys.stderr)
        return 1
    return 0


def cmd_gate(args: argparse.Namespace) -> int:
    failures = gate.run_gate(Path(args.current), Path(args.baseline))
    for f in failures:
        print(f"REGRESSION {f}")
    print("gate: pass" if not failures else f"gate: {len(failures)} regression(s)")
    return 1 if failures else 0


def cmd_validate(args: argparse.Namespace) -> int:
    root, bad = Path(args.data_root), 0
    for s in ALL:
        p = manifest_path(root, s.name)
        if not p.exists():
            continue
        try:
            items = load_manifest(p)
            missing = [
                f"{it['id']}:{k}"
                for it in items
                for k, v in it.items()
                if isinstance(v, str)
                and k in {"image", "audio", "mouth_series", "frames", "series"}
                and not (Path(it["_base"]) / v).exists()
            ]
            print(f"{s.name}: {len(items)} items" + (f", {len(missing)} missing files" if missing else ""))
            bad += bool(missing)
        except ManifestError as e:
            print(f"{s.name}: {e}")
            bad += 1
    return 1 if bad else 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="eval_harness")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--suite", default="all", choices=["all", "smoke", *BY_NAME])
    r.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    r.add_argument("--out", default=str(REPO_ROOT / "eval" / "out"))
    r.add_argument("--report", default=str(REPO_ROOT / "docs" / "EVAL_REPORT.md"))
    r.add_argument("--config", help="JSON with operating thresholds per system")
    r.set_defaults(fn=cmd_run)
    g = sub.add_parser("gate")
    g.add_argument("--current", default=str(REPO_ROOT / "eval" / "out" / "results.json"))
    g.add_argument("--baseline", default=str(REPO_ROOT / "eval" / "baselines" / "baseline.json"))
    g.set_defaults(fn=cmd_gate)
    v = sub.add_parser("validate")
    v.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    v.set_defaults(fn=cmd_validate)
    args = ap.parse_args(argv)
    return args.fn(args)
