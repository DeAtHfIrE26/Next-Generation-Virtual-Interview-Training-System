"""FastAPI application factory."""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from interview_api.db import Base, engine
from interview_api.routers import (
    admin,
    auth,
    avatar,
    billing,
    consent,
    enrollment,
    health,
    privacy,
    reports,
    sessions,
)
from interview_api.security import CSRF_HEADER
from interview_api.settings import get_settings

CSRF_EXEMPT_PREFIXES = ("/billing/webhooks/",)


def create_app(*, create_tables: bool = True) -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title="AI Interview Coach API",
        version="0.1.0",
        docs_url=None if settings.production else "/docs",
        redoc_url=None,
    )

    @app.middleware("http")
    async def guard(request: Request, call_next):
        # Body size limit (Content-Length is required for bodies; chunked uploads are refused).
        if request.method in ("POST", "PUT", "PATCH"):
            length = request.headers.get("content-length")
            if length is None and request.headers.get("transfer-encoding"):
                return JSONResponse({"detail": "content-length required"}, status_code=411)
            if length and int(length) > settings.max_body_bytes:
                return JSONResponse({"detail": "request too large"}, status_code=413)
        # CSRF: state-changing requests must carry a custom header, which a cross-site form or
        # simple request cannot add without a CORS preflight (and CORS is not enabled).
        if (
            request.method in ("POST", "PUT", "PATCH", "DELETE")
            and not request.url.path.startswith(CSRF_EXEMPT_PREFIXES)
            and request.headers.get(CSRF_HEADER) != "1"
        ):
            return JSONResponse({"detail": "missing CSRF header"}, status_code=403)
        response = await call_next(request)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
        response.headers.setdefault("Cache-Control", "no-store")
        return response

    for r in (
        health.router,
        auth.router,
        consent.router,
        enrollment.router,
        sessions.router,
        reports.router,
        privacy.router,
        avatar.router,
    ):
        app.include_router(r)
    app.include_router(billing.router)
    app.include_router(admin.router)
    try:
        from interview_api import observability

        observability.install(app)
    except ImportError:
        pass
    if create_tables:
        Base.metadata.create_all(engine())
    return app


app = create_app(create_tables=not get_settings().production)
