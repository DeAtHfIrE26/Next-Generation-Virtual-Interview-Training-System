import subprocess
import sys
from pathlib import Path

API = Path(__file__).resolve().parents[1]


def test_migrations_apply_and_match_models(tmp_path):
    env = {"DATABASE_URL": f"sqlite:///{tmp_path}/m.db", "PATH": "/usr/bin:/bin"}
    for cmd in (["upgrade", "head"], ["check"], ["downgrade", "base"]):
        r = subprocess.run(
            [sys.executable, "-m", "alembic", *cmd], cwd=API, env=env, capture_output=True, text=True
        )
        assert r.returncode == 0, r.stderr
