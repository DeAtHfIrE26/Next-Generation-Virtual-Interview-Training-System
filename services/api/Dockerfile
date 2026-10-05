# Build from the repository root:  docker build -f services/api/Dockerfile -t interview-api .
FROM python:3.11-slim AS build
COPY --from=ghcr.io/astral-sh/uv:0.8 /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
WORKDIR /app
COPY pyproject.toml uv.lock ./
COPY packages/core packages/core
COPY services/api services/api
COPY eval eval
RUN uv sync --frozen --no-dev --package interview-api --extra postgres --extra gcp --extra llm --extra observability

FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PATH="/app/.venv/bin:$PATH" APP_ENV=production
RUN useradd --system --uid 10001 app
WORKDIR /app
COPY --from=build --chown=app /app /app
USER app
WORKDIR /app/services/api
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s CMD python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8000/health')"
# Migrations run before the server starts; Cloud Run runs one revision at a time per deploy.
CMD ["sh", "-c", "alembic upgrade head && exec uvicorn interview_api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
