# No mocks, fakes or randomness in scoring or question logic (DoD #3)

Audit run: 2026-10-06T20:45Z on commit 3e0f5b1 (earlier runs: f1a2634; first run 2026-10-06T08:27Z on 2bb9a84). Re-run with the commands below.

## 1. `Math.random` in the web app (vendored libraries excluded)

```
$ grep -rn "Math.random" apps/web --include=*.ts --include=*.tsx --include=*.js --include=*.mjs | grep -v "node_modules\|public/vendor\|public/vision\|\.next"
apps/web/lib/voice/realtime.ts:112:    const delay = Math.min(8000, 400 * 2 ** (this.attempts - 1)) + Math.random() * 250; // jitter avoids thundering herd
```

Justification: reconnect backoff jitter for the WebSocket (spreads reconnects after an outage). It never touches questions or scores.

## 2. Randomness in Python product code

```
$ grep -rnE "\bimport random\b|random\.(random|uniform|choice|randint|shuffle|sample)|np\.random" packages/core/src services/api/src --include=*.py
packages/core/src/interview_core/report/__init__.py:5:prototype filled missing series with ``random.uniform``). Integrity notices are reported
packages/core/src/interview_core/codeexec/challenges.py:6:import random
```

Justification:

- `report/__init__.py:5` is a docstring recording that the prototype used `random.uniform`. That code path was removed.
- `codeexec/challenges.py` breaks ties among coding challenges that are equally close to the target difficulty, using `random.Random(session_id)`. The result is deterministic per session, so a reload shows the same challenge. Challenge selection is neither question generation (every spoken question comes from the LLM agent) nor scoring.

Also checked: `grep -rn "secrets\.SystemRandom" packages/core/src services/api/src` finds `voice/prompts.py` and `face/liveness.py`. They generate unpredictable voice-enrolment phrases (E3) and liveness challenge steps (E2), which are anti-replay measures, so they are intentionally random. Neither is scoring or question logic.

## 3. Mock, fake, stub or dummy objects in product code

```
$ grep -rniE "\b(mock|fake|stub|dummy)\b" packages/core/src services/api/src apps/web/app apps/web/lib apps/web/components --include=*.py --include=*.ts --include=*.tsx
packages/core/src/interview_core/agent/prompts.py:27:PLANNER_SYSTEM = f"""You are a senior interviewer preparing a realistic mock interview. Before the interview starts you
packages/core/src/interview_core/agent/prompts.py:53:INTERVIEWER_SYSTEM = f"""You are a skilled, human-sounding interviewer conducting a live, spoken mock interview. Your words are
packages/core/src/interview_core/agent/prompts.py:183:    "You are a human interviewer in a live, spoken mock interview. Write exactly what you say next, "
apps/web/lib/siteGate.test.ts:18:    process.env.SITE_PASSWORD = "dummy-pass";
apps/web/lib/siteGate.test.ts:23:    expect(proxy(req(`Basic ${btoa("anyone:dummy-pass")}`)).status).toBe(200);
```

Justification:

- "mock interview" is the product domain wording inside the LLM prompts.
- `siteGate.test.ts` is a unit test.

Test doubles exist only under test directories:

- `services/api/tests/fake_llm.py` scripts LLM replies for API tests. The speech models are still the real ones.
- `packages/core/tests/test_agent.py` scripts providers to exercise validation and fallback, and `packages/core/tests/fakes.py` holds test-only stand-ins for vendor clients.
- `packages/core/tests/fixtures/legacy_question_bank.json` is the old bank, kept only to assert that no generated question repeats it.

## 4. Fallbacks are flagged, never silent

If every LLM provider fails, the agent emits a flagged backup question. Before that, when the model replies but every full-context attempt fails validation, a short focused LLM call writes the question instead (D19); it is still model-written and checked, and it is labelled in the evidence transcripts. The backup question is flagged as follows:

- `Turn.emergency = True`, a WARNING log line, and an audit or metering record;
- a badge in the room UI and a note in the report;
- `E2E_REQUIRE_LLM=1` fails the E2E on any backup question, and `eval/agent_mock_interviews.py` fails a transcript that contains one.

Scoring has an offline heuristic path. It is labelled "offline scoring" for each answer in the report and recorded as `method: heuristic`. When an LLM is configured but slow, those answers are rescored by the LLM in the background and the report says so until it is done (D21); the real-LLM E2E fails if any answer is left with offline scoring.
