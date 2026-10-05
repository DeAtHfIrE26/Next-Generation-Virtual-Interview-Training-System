# AI Interview Coach

**Practise the interview before the interview.** An adaptive AI interviewer asks questions tailored to your role, seniority and resume, listens to your spoken answers, and gives feedback you can check: every score points to the words you said.

Patent pending (Indian application 202541122226). For candidates practising their own interviews, not for hiring decisions.

<!-- DEMO: replace with recordings -->
| Interview room | Report |
|---|---|
| ![Interview room demo](docs/media/demo-room.gif) | ![Report demo](docs/media/demo-report.gif) |
| _GIF placeholder: avatar asks a question, candidate answers, live captions_ | _GIF placeholder: rubric chart, evidence quotes, share link_ |

## What it does

- **Adaptive interviewer:** questions conditioned on role, seniority, resume and job description. Difficulty moves with each answer, with follow-ups when an answer is thin. Every LLM output is schema-validated, with a deterministic question bank as fallback, so the session always works.
- **Explainable feedback:** relevance, structure (STAR), depth, communication and technical accuracy, each with machine-verified quotes from your answer. Pace, pauses and filler words come with timestamps. Scores are labelled **experimental** until validated against human raters.
- **Multimodal checks** (patented mechanisms):
  - **face verification** with a randomised liveness challenge
  - **voice verification** from prompted phrases
  - **lip-sync verification**: does the mouth movement match the speech over time?
  - **gaze tracking**
  - **phone and second-person detection**
  - **coding challenges** with hidden tests
- **Privacy by design:**
  - camera analysis runs in the browser
  - biometrics are stored only as encrypted templates (Cloud KMS)
  - resume contact details are removed before any AI processing
  - one-click export and deletion
  - no emotion or personality inference, ever
- **Business-ready:** Stripe and Razorpay subscriptions, plan limits and per-session cost caps, B2B seats for colleges and bootcamps, an admin cost/quality dashboard, observability, and infrastructure as code.

## Quick start (local)

```bash
cp .env.example .env
python -c "import os,base64;print(base64.b64encode(os.urandom(32)).decode())"   # paste into TEMPLATE_KEK_BASE64
docker compose up --build          # web on :3000, API on :8000
```

Without Docker:

```bash
uv sync --all-packages                                   # Python 3.11+
TEMPLATE_KEK_BASE64=... uv run uvicorn interview_api.main:app --reload --port 8000
cd apps/web && npm ci && API_BASE_URL=http://localhost:8000 npm run dev
```

Offline mode (no vendor keys) gives the full flow with the question bank and automatic feedback. Add `LLM_PROVIDER`, `ASR_PROVIDER`, `TTS_PROVIDER` and licensed face/voice models to enable the rest (see `.env.example`).

## Quality gates

```bash
make check        # secret scan, ruff, 128 Python tests, legacy compile
make web-test     # eslint, typecheck, vitest
make e2e          # Playwright: full interview in Chromium with fake camera/mic
make eval         # evaluation harness -> docs/EVAL_REPORT.md, then: uv run python -m eval_harness gate
```

CI runs all of these on every push. Accuracy numbers are published **only** in [`docs/EVAL_REPORT.md`](docs/EVAL_REPORT.md), and only when measured on a real, consented dataset. Today every suite reads "not measured"; see [`eval/DATA_COLLECTION.md`](eval/DATA_COLLECTION.md) for what to collect.

## Repository

| Path | |
|---|---|
| `apps/web` | Next.js app (Vercel) |
| `services/api` | FastAPI service (Cloud Run) |
| `packages/core` | Patented algorithms, pure and tested; `legacy/` sub-package reproduces the prototype exactly |
| `eval` | Evaluation harness and regression gate |
| `infra/terraform` | GCP and Vercel infrastructure |
| `legacy/` | Original research prototypes (frozen, not shipped) |

## Documentation

- [Plan](docs/PLAN.md), [Claim map](docs/CLAIM_MAP.md), [Architecture](docs/ARCHITECTURE.md)
- [Evaluation report](docs/EVAL_REPORT.md), [Load test](docs/LOAD_TEST.md), [Rubric](docs/RUBRIC.md)
- [Compliance](COMPLIANCE.md), [Model cards](MODEL_CARDS/), [Privacy notice draft](docs/legal/privacy-policy-draft.md)
- [Licence decision](docs/LICENSE_DECISION.md): no licence has been chosen yet; owner decision pending.

## Credits

Based on research by Dr. Kopperundevi N, Patel Kashyap Kalpeshkumar, Goditi Nishanth Sai Ram and Danaboina Venkata Prabhave (Vellore Institute of Technology), ICCCNT 2025.
