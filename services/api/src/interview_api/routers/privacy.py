"""Data-subject rights: export everything we hold, delete the account. Retention purge job."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from fastapi import APIRouter, Depends, Request, Response
from sqlalchemy.orm import Session

from interview_api.db import get_db
from interview_api.models import (
    AuditLog,
    AuthSession,
    BiometricTemplate,
    Challenge,
    Consent,
    InterviewSession,
    UsageEvent,
    User,
)
from interview_api.security import current_user, end_session
from interview_api.settings import get_settings

router = APIRouter(prefix="/privacy", tags=["privacy"])


@router.get("/export")
def export(user: User = Depends(current_user), db: Session = Depends(get_db)) -> dict:
    return {
        "exported_at": datetime.now(UTC).isoformat(),
        "account": {
            "id": user.id,
            "email": user.email,
            "name": user.name,
            "plan": user.plan,
            "created_at": user.created_at.isoformat(),
        },
        "consents": [
            {"kind": c.kind, "granted": c.granted, "version": c.version, "at": c.created_at.isoformat()}
            for c in db.query(Consent).filter_by(user_id=user.id).order_by(Consent.created_at)
        ],
        "biometric_templates": [
            {
                "kind": t.kind,
                "model": t.model_id,
                "created_at": t.created_at.isoformat(),
                "expires_at": t.expires_at.isoformat(),
                "note": "Stored only as an encrypted numeric template; it cannot be turned back into an image or recording.",
            }
            for t in db.query(BiometricTemplate).filter_by(user_id=user.id)
        ],
        "sessions": [
            {
                "id": s.id,
                "role": s.role,
                "created_at": s.created_at.isoformat(),
                "status": s.status,
                "state": s.state,
                "signals": s.signals,
                "integrity": s.integrity,
                "report": s.report,
            }
            for s in db.query(InterviewSession).filter_by(user_id=user.id)
        ],
        "usage": [
            {"kind": u.kind, "quantity": u.quantity, "at": u.created_at.isoformat()}
            for u in db.query(UsageEvent).filter_by(user_id=user.id)
        ],
    }


@router.delete("/account")
def delete_account(
    request: Request, response: Response, user: User = Depends(current_user), db: Session = Depends(get_db)
) -> dict:
    uid = user.id
    end_session(db, request, response)
    db.delete(user)  # cascades: sessions, templates, consents, usage, auth sessions, challenges
    db.add(AuditLog(actor_id=uid, action="delete_account"))
    db.commit()
    return {"deleted": True}


def purge_expired(db: Session, now: datetime | None = None) -> dict[str, int]:
    """Run daily (``python -m interview_api.jobs purge``)."""
    now = now or datetime.now(UTC)
    s = get_settings()
    out = {
        "templates": db.query(BiometricTemplate).filter(BiometricTemplate.expires_at < now).delete(),
        "challenges": db.query(Challenge).filter(Challenge.expires_at < now - timedelta(days=1)).delete(),
        "auth_sessions": db.query(AuthSession).filter(AuthSession.expires_at < now).delete(),
        "sessions": db.query(InterviewSession)
        .filter(InterviewSession.created_at < now - timedelta(days=s.session_retention_days))
        .delete(),
    }
    db.add(AuditLog(actor_id=None, action="retention_purge", target=str(out)))
    db.commit()
    return out
