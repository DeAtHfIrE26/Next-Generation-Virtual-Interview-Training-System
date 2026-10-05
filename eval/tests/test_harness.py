import json

import pytest
from eval_harness import gate
from eval_harness.base import ManifestError, load_manifest
from eval_harness.cli import main, run_smoke


def test_manifest_requires_consent(tmp_path):
    p = tmp_path / "manifest.jsonl"
    p.write_text(json.dumps({"id": "x", "audio": "a.wav"}) + "\n")
    with pytest.raises(ManifestError, match="consent_id"):
        load_manifest(p)


def test_manifest_rejects_bad_json_and_empty(tmp_path):
    p = tmp_path / "manifest.jsonl"
    p.write_text("{not json}\n")
    with pytest.raises(ManifestError, match="invalid JSON"):
        load_manifest(p)
    p.write_text("\n")
    with pytest.raises(ManifestError, match="empty"):
        load_manifest(p)


def test_smoke_runs_every_local_pipeline_and_hides_numbers():
    results = run_smoke({})
    assert results, "smoke produced nothing"
    assert all(r.status == "smoke" for r in results), [(r.suite, r.status, r.notes) for r in results]
    assert all(s.metrics == {} for r in results for s in r.systems)


def test_full_run_without_data_reports_not_measured(tmp_path):
    out, rep = tmp_path / "out", tmp_path / "EVAL.md"
    assert main(["run", "--data-root", str(tmp_path / "none"), "--out", str(out), "--report", str(rep)]) == 0
    text = rep.read_text()
    assert "NOT MEASURED" in text and "Smoke runs" in text
    payload = json.loads((out / "results.json").read_text())
    assert all(r["status"] in {"no_data", "not_configured"} for r in payload["results"])


def _payload(eer=None, wer=None, p95=None):
    systems = {"system": "s", "metrics": {}}
    if eer is not None:
        systems["metrics"]["eer"] = eer
    if wer is not None:
        systems["metrics"]["wer"] = wer
    if p95 is not None:
        systems["metrics"]["p95_ms"] = p95
    return {"results": [{"suite": "x", "status": "measured", "systems": [systems]}]}


def test_gate_tolerances():
    assert gate.compare(_payload(eer=0.054), _payload(eer=0.05)) == []
    assert gate.compare(_payload(eer=0.06), _payload(eer=0.05))
    assert gate.compare(_payload(p95=1090), _payload(p95=1000)) == []
    assert gate.compare(_payload(p95=1200), _payload(p95=1000))
    assert gate.compare(_payload(wer=0.05), _payload(wer=0.10)) == []  # improvement passes
