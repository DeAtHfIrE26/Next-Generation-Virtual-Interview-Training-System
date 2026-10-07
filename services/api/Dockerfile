# syntax=docker/dockerfile:1
# Build from the repository root:  docker build -f services/api/Dockerfile -t interview-api .
# Behind a TLS-intercepting proxy, pass its CA bundle:  --secret id=extra_ca,src=/path/ca.pem
FROM python:3.11-slim AS build
# uv from PyPI, not ghcr.io, so the image builds on networks that block ghcr.io (DECISIONS D5).
RUN --mount=type=secret,id=extra_ca,required=false \
    if [ -s /run/secrets/extra_ca ]; then export PIP_CERT=/run/secrets/extra_ca; fi; \
    pip install --no-cache-dir "uv==0.8.*"
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
WORKDIR /app
COPY pyproject.toml uv.lock ./
COPY packages/core packages/core
COPY services/api services/api
COPY eval eval
RUN --mount=type=secret,id=extra_ca,required=false \
    if [ -s /run/secrets/extra_ca ]; then export SSL_CERT_FILE=/run/secrets/extra_ca; fi; \
    uv sync --frozen --no-dev --package interview-api --extra postgres --extra gcp --extra llm --extra observability

FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PATH="/app/.venv/bin:$PATH" APP_ENV=production MODELS_DIR=/models
RUN useradd --system --uid 10001 app && mkdir -p /models && chown app /models
WORKDIR /app
COPY --from=build --chown=app /app /app
USER app
WORKDIR /app/services/api
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s CMD python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8000/health')"
# Migrations run before the server starts; Cloud Run runs one revision at a time per deploy.
CMD ["sh", "-c", "alembic upgrade head && exec uvicorn interview_api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
