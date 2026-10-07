"""Run a submission against all test cases.

General-purpose languages run only in a Judge0 sandbox (self-hosted Judge0 CE, or the hosted
API with a key); candidate code is never executed in the API process. SQL challenges may also
run in an in-memory SQLite database inside a step-limited, extension-free connection.
"""

from __future__ import annotations

import os
import sqlite3
from dataclasses import dataclass, field

import httpx

from interview_core.codeexec.challenges import Challenge

LANGUAGE_IDS = {"python": 71, "javascript": 63, "java": 62, "cpp": 54, "sql": 82}  # Judge0 CE ids
MAX_SOURCE_BYTES = 64_000


@dataclass
class TestOutcome:
    passed: bool
    hidden: bool
    status: str
    stdout: str | None = None  # withheld for hidden tests
    expected: str | None = None
    time_s: float | None = None


@dataclass
class RunResult:
    challenge_id: str
    language: str
    passed: int
    total: int
    outcomes: list[TestOutcome] = field(default_factory=list)
    error: str | None = None
    executor: str = ""

    def summary(self) -> str:
        if self.error:
            return f"Could not run: {self.error}"
        return f"{self.passed}/{self.total} tests passed"


def normalise_output(s: str) -> str:
    return "\n".join(line.rstrip() for line in s.strip().splitlines())


class Judge0Client:
    def __init__(self, base_url: str, rapidapi_key: str = "", timeout_s: float = 30.0):
        self.base_url, self.timeout_s = base_url.rstrip("/"), timeout_s
        self.headers = {}
        if rapidapi_key:
            self.headers = {"x-rapidapi-key": rapidapi_key, "x-rapidapi-host": httpx.URL(self.base_url).host}

    @classmethod
    def from_env(cls) -> Judge0Client | None:
        url = os.getenv("JUDGE0_URL", "")
        return cls(url, os.getenv("JUDGE0_RAPIDAPI_KEY", "")) if url else None

    def run(self, source: str, language_id: int, stdin: str) -> dict:
        r = httpx.post(
            f"{self.base_url}/submissions",
            params={"base64_encoded": "false", "wait": "true"},
            headers=self.headers,
            timeout=self.timeout_s,
            json={
                "source_code": source,
                "language_id": language_id,
                "stdin": stdin,
                "cpu_time_limit": 2,
                "wall_time_limit": 5,
                "memory_limit": 128000,
            },
        )
        r.raise_for_status()
        return r.json()


def run_sql_locally(setup: str, query: str, max_steps: int = 2_000_000) -> str:
    conn = sqlite3.connect(":memory:")
    try:
        conn.enable_load_extension(False)
        steps = {"n": 0}

        def guard() -> int:
            steps["n"] += 1
            return 1 if steps["n"] > max_steps // 1000 else 0

        conn.set_progress_handler(guard, 1000)
        conn.executescript(setup)
        statements = [s for s in query.strip().rstrip(";").split(";") if s.strip()]
        if len(statements) != 1 or not statements[0].lstrip().lower().startswith(("select", "with")):
            raise ValueError("submit exactly one SELECT query")
        rows = conn.execute(statements[0]).fetchall()
        return "\n".join("|".join("" if v is None else str(v) for v in row) for row in rows)
    finally:
        conn.close()


def grade_submission(ch: Challenge, language: str, source: str, judge: Judge0Client | None) -> RunResult:
    res = RunResult(ch.id, language, 0, len(ch.tests))
    if language not in ch.languages:
        res.error = f"{language} is not available for this challenge"
        return res
    if len(source.encode()) > MAX_SOURCE_BYTES:
        res.error = "submission too large"
        return res
    for t in ch.tests:
        try:
            if language == "sql" and judge is None:
                res.executor = "sqlite-local"
                out, status, secs = run_sql_locally(ch.setup, source), "ok", None
            elif judge is None:
                res.error = "code execution service is not configured"
                return res
            else:
                res.executor = "judge0"
                payload = (ch.setup + "\n" + source) if language == "sql" else source
                data = judge.run(payload, LANGUAGE_IDS[language], t.stdin)
                status = (data.get("status") or {}).get("description", "unknown")
                out = data.get("stdout") or ""
                if data.get("compile_output"):
                    status = "compilation error"
                secs = float(data["time"]) if data.get("time") else None
        except (sqlite3.Error, ValueError) as e:
            out, status, secs = "", f"error: {e}", None
        except httpx.HTTPError:
            res.error = "code execution service unavailable; try again"
            return res
        ok = normalise_output(out) == normalise_output(t.expected)
        res.passed += ok
        res.outcomes.append(
            TestOutcome(
                ok,
                t.hidden,
                "passed" if ok else status if status != "ok" else "wrong answer",
                None if t.hidden else out[:2000],
                None if t.hidden else t.expected,
                secs,
            )
        )
    return res
