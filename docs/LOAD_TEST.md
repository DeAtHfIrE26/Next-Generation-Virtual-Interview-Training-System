# Load test

`loadtest/run.py` simulates complete candidate sessions: sign up → consent → create session → (next question → answer) × N → finish with a report.

## Run of 2026-10-05

**Setup** (as measured; not a production sizing):
- API: `uvicorn` with 2 workers in a 4-vCPU container.
- Database: local PostgreSQL 16.
- Offline mode: question bank and heuristic evaluation, no LLM, ASR or TTS vendor calls. **Vendor latency is therefore not included.** With an LLM, each `next_question` and `answer` adds the provider's response time (recorded separately as `model_call_seconds` in `/metrics` and in the admin dashboard).
- Rate limits multiplied by 100 (`RATE_LIMIT_MULTIPLIER`), because every virtual user shares one IP.

| Concurrent sessions | Completed | Errors | Throughput | `next_question` p50 / p95 | `answer` p50 / p95 | `register` p50 / p95 | Whole session p50 / p95 |
|---|---|---|---|---|---|---|---|
| 25 | 25 / 25 | 0 | 63.5 req/s | 203 / 307 ms | 224 / 313 ms | 3450 / 4001 ms | 8.7 / 9.2 s |
| 50 | 50 / 50 | 0 | 76.7 req/s | 382 / 693 ms | 419 / 659 ms | 3751 / 4929 ms | 13.7 / 15.2 s |

## Findings

- **Registration bursts are CPU-bound.** Argon2id password hashing is deliberately expensive. Fifty simultaneous sign-ups queue behind it. Real sign-ups are spread out, and Cloud Run adds instances as CPU rises. If launch-day bursts are expected, set a minimum instance count for the launch.
- **Interview turns stay well within interactive limits.** At 50 concurrent sessions on two workers, p95 is under 0.7 s for both question and answer, excluding vendor time. Cloud Run is configured for 40 concurrent requests per instance, so scale-out starts before this load.
- **Not yet measured:** behaviour with real LLM, ASR and TTS providers (their rate limits usually bind first), and a multi-instance run against Cloud SQL. Repeat this test on staging with providers enabled before launch, and record the result here.

## Reproduce

```bash
DATABASE_URL=postgresql+psycopg://... RATE_LIMIT_MULTIPLIER=100 uvicorn interview_api.main:app --workers 2 --port 8200
uv run python loadtest/run.py --base http://127.0.0.1:8200 --users 50 --questions 5
```
