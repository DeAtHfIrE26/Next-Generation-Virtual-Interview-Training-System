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
