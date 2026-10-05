import importlib.util
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]


def test_web_parity_fixture_is_current():
    spec = importlib.util.spec_from_file_location("gen", REPO / "tools/gen_parity_fixtures.py")
    gen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gen)
    committed = json.loads((REPO / "apps/web/lib/__fixtures__/parity.json").read_text())
    fresh = json.loads(json.dumps(gen.build()))
    for a, b in zip(committed["landmarks"], fresh["landmarks"], strict=True):
        assert a["mouth"] == pytest.approx(b["mouth"], rel=1e-9)
        assert a["on_screen"] == b["on_screen"]
    assert committed["visemes"] == fresh["visemes"]
