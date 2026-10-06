# API on Vercel (private preview)

Vercel project settings: root directory `deploy/vercel-api`, framework FastAPI, region `bom1`.

| Variable | Value |
|---|---|
| `API_SHARED_SECRET` | same random value as on the web project |
| `DATABASE_URL` | `sqlite:////tmp/interview.db` for a throwaway preview (**data resets** whenever the function instance is recycled), or a Postgres URL (`postgresql+psycopg://...`) for anything that must persist |
| `APP_ENV` | `preview` (tables are created on start; set `production` only with Alembic migrations applied) |
| `COOKIE_SECURE` | `true` |
| `APP_BASE_URL` | the web app URL |

The web project (root `apps/web`) needs `API_BASE_URL` (this project's URL), `API_SHARED_SECRET`, and optionally `SITE_PASSWORD`.

This is a preview setup. The production path is Cloud Run + Cloud SQL (`infra/terraform/gcp`).

## Current private preview (deployed 2026-10-06)

| | Project | URL | Access |
|---|---|---|---|
| Web | `interview-coach-web` | https://interview-coach-web-eta.vercel.app | Vercel Authentication on all deployments (owner's Vercel login only) |
| API | `interview-coach-api` | https://interview-coach-api.vercel.app | Everything except `/health` requires `API_SHARED_SECRET`, sent only by the web proxy |

Both run in `bom1` (Mumbai), offline mode (question bank, automatic feedback), SQLite in `/tmp` (data resets).
To share with testers: set `SITE_PASSWORD` on the web project and switch Vercel Authentication off, then redeploy.
