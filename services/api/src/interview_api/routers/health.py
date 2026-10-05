"""Liveness/readiness, capabilities and client latency reports."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from sqlalchemy import text
from sqlalchemy.orm import Session

from interview_api import runtime
from interview_api.db import get_db
from interview_api.models import LatencyEvent
from interview_api.ratelimit import limiter

router = APIRouter(tags=["health"])
latency_limit = limiter("latency", capacity=120, per_seconds=60)


@router.get("/health")
def health() -> dict:
    return {"ok": True}


@router.get("/ready")
def ready(db: Session = Depends(get_db)) -> dict:
    db.execute(text("SELECT 1"))
    return {"ok": True, "capabilities": runtime.capabilities()}


class LatencyIn(BaseModel):
    metric: str = Field(pattern=r"^[a-z0-9_]{3,60}$")
    ms: float = Field(ge=0, le=600_000)
    session_id: str | None = Field(default=None, max_length=32)


@router.post("/metrics/latency", dependencies=[Depends(latency_limit)])
def report_latency(body: LatencyIn, db: Session = Depends(get_db)) -> dict:
    db.add(LatencyEvent(metric=body.metric, ms=body.ms, session_id=None))
    db.commit()
    return {"ok": True}
