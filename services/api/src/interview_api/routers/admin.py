"""Admin: per-session cost, model quality and latency dashboards, B2B seats, eval exports."""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import UTC, datetime, timedelta

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from interview_api import metering, runtime
from interview_api.db import get_db
from interview_api.models import AuditLog, InterviewSession, LatencyEvent, LLMCall, Org, UsageEvent, User
from interview_api.security import admin_user
from interview_api.settings import get_settings

router = APIRouter(prefix="/admin", tags=["admin"], dependencies=[Depends(admin_user)])


def _mask(email: str) -> str:
    name, _, domain = email.partition("@")
    return f"{name[:2]}***@{domain}"


@router.get("/overview")
def overview(days: int = Query(30, ge=1, le=365), db: Session = Depends(get_db)) -> dict:
    since = datetime.now(UTC) - timedelta(days=days)
    usage = db.query(UsageEvent).filter(UsageEvent.created_at >= since).all()
    per_session: dict[str, dict] = defaultdict(lambda: {"total": 0, "by_kind": defaultdict(int)})
    by_kind: dict[str, dict] = defaultdict(lambda: {"quantity": 0.0, "cost_micro_usd": 0})
    for u in usage:
        if u.session_id:
            per_session[u.session_id]["total"] += u.cost_micro_usd
            per_session[u.session_id]["by_kind"][u.kind] += u.cost_micro_usd
        k = by_kind[f"{u.kind}|{u.provider}"]
        k["quantity"] += u.quantity
        k["cost_micro_usd"] += u.cost_micro_usd
    sessions = {
        s.id: s for s in db.query(InterviewSession).filter(InterviewSession.id.in_(list(per_session)))
    }
    users = {u.id: u for u in db.query(User).filter(User.id.in_([s.user_id for s in sessions.values()]))}
    rows = sorted(
        (
            {
                "session_id": sid,
                "created_at": sessions[sid].created_at.isoformat() if sid in sessions else None,
                "user": _mask(users[sessions[sid].user_id].email) if sid in sessions else None,
                "plan": users[sessions[sid].user_id].plan if sid in sessions else None,
                "total_micro_usd": v["total"],
                "by_kind": dict(v["by_kind"]),
            }
            for sid, v in per_session.items()
        ),
        key=lambda r: -r["total_micro_usd"],
    )[:200]
    totals = [r["total_micro_usd"] for r in rows]

    calls = db.query(LLMCall).filter(LLMCall.created_at >= since).all()
    quality: dict[str, dict] = {}
    for key in sorted({(c.provider, c.task) for c in calls}):
        cs = [c for c in calls if (c.provider, c.task) == key]
        quality[f"{key[0]}/{key[1]}"] = {
            "calls": len(cs),
            "raw_schema_valid_rate": round(float(np.mean([c.raw_valid for c in cs])), 4),
            "fallback_rate": round(float(np.mean([c.used_fallback for c in cs])), 4),
            "p95_latency_ms": round(float(np.percentile([c.latency_ms for c in cs], 95)), 1),
        }
    lat = db.query(LatencyEvent).filter(LatencyEvent.created_at >= since).all()
    latency = {
        m: {"n": len(v), "p50_ms": float(np.percentile(v, 50)), "p95_ms": float(np.percentile(v, 95))}
        for m in sorted({e.metric for e in lat})
        for v in [[e.ms for e in lat if e.metric == m]]
    }
    return {
        "days": days,
        "sessions": db.query(InterviewSession).filter(InterviewSession.created_at >= since).count(),
        "users": db.query(User).count(),
        "cost": {
            "per_session": rows,
            "by_kind_provider": by_kind,
            "session_p50_micro_usd": float(np.percentile(totals, 50)) if totals else None,
            "session_p95_micro_usd": float(np.percentile(totals, 95)) if totals else None,
            "prices_configured": metering.prices_configured(),
        },
        "llm_quality": quality,
        "latency": latency,
        "capabilities": runtime.capabilities(),
        "flags": {
            "b2b_hiring": get_settings().feature_b2b_hiring,
            "neural_avatar": get_settings().feature_neural_avatar,
        },
    }


@router.get("/export/{suite}", response_class=PlainTextResponse)
def export_manifest(suite: str, db: Session = Depends(get_db)) -> str:
    """JSONL manifests for the evaluation harness (eval/data/<suite>/manifest.jsonl)."""
    if suite == "llm_schema_validity":
        rows = [
            {
                "id": f"call-{c.id}",
                "consent_id": "system",
                "provider": f"{c.provider}:{c.model}",
                "task": c.task,
                "raw_valid": c.raw_valid,
                "delivered_valid": c.delivered_valid,
                "used_fallback": c.used_fallback,
            }
            for c in db.query(LLMCall)
        ]
    elif suite == "latency":
        rows = [
            {"id": f"lat-{e.id}", "consent_id": "system", "metric": e.metric, "ms": e.ms}
            for e in db.query(LatencyEvent)
        ]
    else:
        raise HTTPException(404, "exportable suites: llm_schema_validity, latency")
    return "\n".join(json.dumps(r) for r in rows) + ("\n" if rows else "")


class OrgIn(BaseModel):
    name: str = Field(min_length=2, max_length=200)
    seats: int = Field(ge=1, le=10000)


@router.post("/orgs", status_code=201)
def create_org(body: OrgIn, admin: User = Depends(admin_user), db: Session = Depends(get_db)) -> dict:
    org = Org(name=body.name, seats=body.seats)
    db.add(org)
    db.add(AuditLog(actor_id=admin.id, action="create_org", target=body.name))
    db.commit()
    return {"id": org.id, "name": org.name, "seats": org.seats}


class MemberIn(BaseModel):
    email: str


@router.post("/orgs/{org_id}/members")
def add_member(
    org_id: str, body: MemberIn, admin: User = Depends(admin_user), db: Session = Depends(get_db)
) -> dict:
    org = db.get(Org, org_id)
    if org is None:
        raise HTTPException(404, "org not found")
    used = db.query(User).filter_by(org_id=org.id).count()
    u = db.query(User).filter_by(email=body.email.strip().lower()).first()
    if u is None:
        raise HTTPException(404, "user must sign up first")
    if u.org_id != org.id and used >= org.seats:
        raise HTTPException(409, f"all {org.seats} seats are in use")
    u.org_id, u.plan = org.id, "team"
    db.add(AuditLog(actor_id=admin.id, action="add_member", target=f"{org.id}:{u.id}"))
    db.commit()
    return {
        "org_id": org.id,
        "seats_used": db.query(User).filter_by(org_id=org.id).count(),
        "seats": org.seats,
    }


@router.delete("/orgs/{org_id}/members/{user_id}")
def remove_member(
    org_id: str, user_id: str, admin: User = Depends(admin_user), db: Session = Depends(get_db)
) -> dict:
    u = db.get(User, user_id)
    if u is None or u.org_id != org_id:
        raise HTTPException(404, "member not found")
    u.org_id, u.plan = None, "free"
    db.add(AuditLog(actor_id=admin.id, action="remove_member", target=f"{org_id}:{user_id}"))
    db.commit()
    return {"removed": True}
