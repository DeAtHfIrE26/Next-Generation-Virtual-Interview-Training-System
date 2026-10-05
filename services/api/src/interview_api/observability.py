"""Structured logs, request ids, Prometheus metrics, optional OpenTelemetry traces and Sentry.

Logs are JSON lines with a redaction filter: anything that looks like a credential, an email
address or a phone number is masked before it is written, and request/response bodies are
never logged (they carry answers, resumes and biometric data).
"""

from __future__ import annotations

import json
import logging
import os
import re
import secrets
import sys
import time
from contextvars import ContextVar

from fastapi import FastAPI, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

request_id: ContextVar[str] = ContextVar("request_id", default="-")

REQUESTS = Counter("http_requests_total", "HTTP requests", ["method", "route", "status"])
LATENCY = Histogram(
    "http_request_seconds",
    "HTTP request latency",
    ["method", "route"],
    buckets=(0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30),
)
MODEL_CALLS = Counter("model_calls_total", "Model/provider calls", ["provider", "task", "outcome"])
MODEL_LATENCY = Histogram(
    "model_call_seconds",
    "Model/provider call latency",
    ["provider", "task"],
    buckets=(0.1, 0.25, 0.5, 1, 2, 4, 8, 16, 32),
)

_REDACTIONS = [
    (re.compile(r"(?i)\b(bearer|basic|token)\s+[A-Za-z0-9._~+/=-]{3,}"), r"\1 <REDACTED>"),
    (re.compile(r"\b(?:sk|rk|pk)_(?:live|test)_[A-Za-z0-9]{8,}\b"), "<REDACTED:key>"),
    (re.compile(r"\bsk-(?:ant-|proj-)?[A-Za-z0-9_-]{16,}\b"), "<REDACTED:key>"),
    (re.compile(r"\bhf_[A-Za-z0-9]{20,}\b"), "<REDACTED:key>"),
    (
        re.compile(
            r"(?i)(authorization|cookie|x-api-key|password|secret|token)([\"']?\s*[:=]\s*[\"']?)[^\s,\"'}]+"
        ),
        r"\1\2<REDACTED>",
    ),
    (re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"), "<REDACTED:email>"),
    (re.compile(r"(?<!\d)\+?\d[\d\s-]{9,}\d(?!\d)"), "<REDACTED:phone>"),
]


def redact(text: str) -> str:
    for pattern, repl in _REDACTIONS:
        text = pattern.sub(repl, text)
    return text


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": round(record.created, 3),
            "level": record.levelname,
            "logger": record.name,
            "msg": redact(record.getMessage()),
            "request_id": request_id.get(),
        }
        for key in ("route", "status", "duration_ms", "method"):
            if hasattr(record, key):
                payload[key] = getattr(record, key)
        if record.exc_info:
            payload["exc"] = redact(self.formatException(record.exc_info))
        return json.dumps(payload)


def configure_logging() -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(os.getenv("LOG_LEVEL", "INFO"))
    for noisy in ("uvicorn.access",):
        logging.getLogger(noisy).handlers[:] = []
        logging.getLogger(noisy).propagate = False


def record_model_call(provider: str, task: str, ok: bool, fallback: bool, seconds: float) -> None:
    outcome = "fallback" if fallback else ("ok" if ok else "invalid")
    MODEL_CALLS.labels(provider, task, outcome).inc()
    MODEL_LATENCY.labels(provider, task).observe(seconds)


def _route(request: Request) -> str:
    r = request.scope.get("route")
    return getattr(r, "path", "unmatched")


def install(app: FastAPI) -> None:
    configure_logging()
    log = logging.getLogger("interview_api.http")

    @app.middleware("http")
    async def observe(request: Request, call_next):
        rid = request.headers.get("x-request-id", "")[:64] or secrets.token_hex(8)
        token = request_id.set(rid)
        start = time.perf_counter()
        status = 500
        try:
            response = await call_next(request)
            status = response.status_code
            response.headers["x-request-id"] = rid
            return response
        finally:
            dur = time.perf_counter() - start
            route = _route(request)
            REQUESTS.labels(request.method, route, str(status)).inc()
            LATENCY.labels(request.method, route).observe(dur)
            log.info(
                "request",
                extra={
                    "route": route,
                    "status": status,
                    "method": request.method,
                    "duration_ms": round(dur * 1000, 1),
                },
            )
            request_id.reset(token)

    @app.get("/metrics", include_in_schema=False)
    def metrics(request: Request) -> Response:
        expected = os.getenv("METRICS_TOKEN", "")
        if expected and request.headers.get("authorization") != f"Bearer {expected}":
            return Response(status_code=401)
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    _maybe_tracing(app)
    _maybe_sentry()


def _maybe_tracing(app: FastAPI) -> None:
    if not os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT"):
        return
    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
    except ImportError:
        logging.getLogger(__name__).warning("OTEL endpoint set but opentelemetry extras not installed")
        return
    provider = TracerProvider(resource=Resource.create({"service.name": "interview-api"}))
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    trace.set_tracer_provider(provider)
    FastAPIInstrumentor.instrument_app(app, excluded_urls="health,ready,metrics")


def _scrub_event(event, _hint):
    req = event.get("request") or {}
    for k in ("data", "cookies", "headers", "query_string"):
        req.pop(k, None)
    return event


def _maybe_sentry() -> None:
    dsn = os.getenv("SENTRY_DSN")
    if not dsn:
        return
    try:
        import sentry_sdk
    except ImportError:
        logging.getLogger(__name__).warning("SENTRY_DSN set but sentry-sdk not installed")
        return
    sentry_sdk.init(
        dsn=dsn,
        send_default_pii=False,
        traces_sample_rate=float(os.getenv("SENTRY_TRACES", "0.05")),
        before_send=_scrub_event,
        environment=os.getenv("APP_ENV", "development"),
    )
