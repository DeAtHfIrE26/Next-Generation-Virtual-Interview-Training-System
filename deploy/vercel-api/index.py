"""Vercel entrypoint: the FastAPI app from services/api (see pyproject.toml)."""

from interview_api.main import app

__all__ = ["app"]
