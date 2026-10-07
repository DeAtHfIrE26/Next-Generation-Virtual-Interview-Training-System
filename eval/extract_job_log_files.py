"""Recover files printed into a CI job log between ``BEGIN_FILE name`` and ``END_FILE name``.

The agent-evidence workflow prints each transcript into its job log as well as uploading it as an
artifact, so the evidence can be rebuilt where artifact downloads are not available.

Usage:
  uv run python eval/extract_job_log_files.py --out docs/evidence/questions LOG [LOG ...]

LOG is a saved job log (plain text, or the JSON a log API returned with a ``logs_content`` field).
A block named ``*.b64`` holds base64 (binary evidence such as video); it is decoded and written
without the suffix.
"""

from __future__ import annotations

import argparse
import base64
import json
import re
from pathlib import Path

TS = re.compile(r"^\ufeff?\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d+Z ?")  # GitHub starts log chunks with a BOM
SAFE = re.compile(r"^[A-Za-z0-9._-]+$")


def log_text(path: Path) -> str:
    raw = path.read_text(encoding="utf-8", errors="replace")
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        return raw
    if isinstance(obj, dict):
        if "logs_content" in obj:
            return str(obj["logs_content"])
        if isinstance(obj.get("logs"), list):
            return "\n".join(str(j.get("logs_content", "")) for j in obj["logs"])
    return raw


def extract(text: str) -> dict[str, str]:
    files: dict[str, str] = {}
    name: str | None = None
    buf: list[str] = []
    for line in text.splitlines():
        line = TS.sub("", line)
        if line.startswith("BEGIN_FILE "):
            name, buf = line.removeprefix("BEGIN_FILE ").strip(), []
        elif line.startswith("END_FILE ") and name is not None:
            if SAFE.match(name):
                files[name] = "\n".join(buf).rstrip("\n") + "\n"
            name = None
        elif name is not None:
            buf.append(line)
    return files


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("logs", nargs="+", type=Path)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    n = 0
    for p in a.logs:
        for name, body in extract(log_text(p)).items():
            if name.endswith(".b64"):
                data = base64.b64decode("".join(body.split()))
                (a.out / name.removesuffix(".b64")).write_bytes(data)
                name, size = name.removesuffix(".b64"), len(data)
            else:
                (a.out / name).write_text(body, encoding="utf-8")
                size = len(body)
            n += 1
            print(f"{p.name}: {name} ({size} bytes)")
    print(f"{n} files written to {a.out}")
    return 0 if n else 1


if __name__ == "__main__":
    raise SystemExit(main())
