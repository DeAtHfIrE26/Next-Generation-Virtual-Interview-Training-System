"""Concurrent interview-session load test (asyncio + httpx; no extra dependencies).

Each virtual user signs up, consents, creates a session, answers every question, and
finishes with a report. Reports per-endpoint latency (p50/p95/max), error counts and
whole-session time.

    uv run python loadtest/run.py --base http://127.0.0.1:8000 --users 25 --questions 5

Run the API with RATE_LIMIT_MULTIPLIER raised (all users share one IP here).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
import uuid
from collections import defaultdict

import httpx
import numpy as np

ANSWER = (
    "At my previous job I led the migration of our billing service. I profiled the slow queries, "
    "added indexes and rewrote the batch job. As a result the nightly run dropped from six hours to forty minutes."
)
WORDS = [{"word": w, "start": i * 0.35, "end": i * 0.35 + 0.3} for i, w in enumerate(ANSWER.split())]


async def user(base: str, questions: int, lat: dict, errors: dict, session_times: list) -> None:
    async with httpx.AsyncClient(base_url=base, headers={"x-ic-csrf": "1"}, timeout=60) as c:

        async def call(name: str, method: str, url: str, **kw):
            t = time.perf_counter()
            r = await c.request(method, url, **kw)
            lat[name].append((time.perf_counter() - t) * 1000)
            if r.status_code >= 400:
                errors[f"{name}:{r.status_code}"] += 1
                raise RuntimeError(f"{name} -> {r.status_code}")
            return r.json()

        t0 = time.perf_counter()
        try:
            await call(
                "register",
                "POST",
                "/auth/register",
                json={
                    "email": f"load-{uuid.uuid4().hex[:12]}@example.com",
                    "password": "load test password",
                    "accept_terms": True,
                },
            )
            await call("consent", "POST", "/consent", json={"kind": "data_processing", "granted": True})
            sid = (
                await call(
                    "create_session",
                    "POST",
                    "/sessions",
                    data={"role": "Backend Software Engineer", "length": str(questions)},
                )
            )["id"]
            for _ in range(questions * 2):
                q = await call("next_question", "POST", f"/sessions/{sid}/next")
                if q.get("done"):
                    break
                await call(
                    "answer", "POST", f"/sessions/{sid}/answer", json={"transcript": ANSWER, "words": WORDS}
                )
            await call("finish", "POST", f"/sessions/{sid}/finish")
            session_times.append((time.perf_counter() - t0) * 1000)
        except RuntimeError:
            pass


async def main(args) -> dict:
    lat: dict[str, list[float]] = defaultdict(list)
    errors: dict[str, int] = defaultdict(int)
    session_times: list[float] = []
    t = time.perf_counter()
    await asyncio.gather(
        *(user(args.base, args.questions, lat, errors, session_times) for _ in range(args.users))
    )
    wall = time.perf_counter() - t
    pct = lambda v, p: round(float(np.percentile(v, p)), 1)  # noqa: E731
    return {
        "users": args.users,
        "questions_per_session": args.questions,
        "wall_s": round(wall, 1),
        "completed_sessions": len(session_times),
        "errors": dict(errors),
        "requests": sum(len(v) for v in lat.values()),
        "throughput_rps": round(sum(len(v) for v in lat.values()) / wall, 1),
        "endpoints": {
            k: {"n": len(v), "p50_ms": pct(v, 50), "p95_ms": pct(v, 95), "max_ms": round(max(v), 1)}
            for k, v in sorted(lat.items())
        },
        "session_ms": {"p50": pct(session_times, 50), "p95": pct(session_times, 95)}
        if session_times
        else None,
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--users", type=int, default=25)
    ap.add_argument("--questions", type=int, default=5)
    print(json.dumps(asyncio.run(main(ap.parse_args())), indent=2))
