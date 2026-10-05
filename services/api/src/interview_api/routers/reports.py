"""Reports and revocable share links (E9)."""

from __future__ import annotations

import secrets
from datetime import UTC, datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from interview_api.db import get_db
from interview_api.models import InterviewSession, User
from interview_api.ratelimit import limiter
from interview_api.routers.sessions import _own
from interview_api.security import current_user, token_hash
from interview_api.settings import get_settings

router = APIRouter(tags=["reports"])
share_limit = limiter("shared", capacity=30, per_seconds=60, per_session=False)


@router.get("/reports/{session_id}")
def get_report(session_id: str, user: User = Depends(current_user), db: Session = Depends(get_db)) -> dict:
    s = _own(db, user, session_id)
    if s.report is None:
        raise HTTPException(404, "report not ready; finish the session first")
    return s.report


class ShareIn(BaseModel):
    days: int = Field(default=14, ge=1, le=90)


@router.post("/reports/{session_id}/share")
def share(
    session_id: str, body: ShareIn, user: User = Depends(current_user), db: Session = Depends(get_db)
) -> dict:
    s = _own(db, user, session_id)
    if s.report is None:
        raise HTTPException(409, "finish the session before sharing")
    token = secrets.token_urlsafe(24)
    s.share_token_hash = token_hash(token)
    s.share_expires_at = datetime.now(UTC) + timedelta(days=body.days)
    db.commit()
    return {
        "url": f"{get_settings().app_base_url}/share/{token}",
        "expires_at": s.share_expires_at.isoformat(),
    }


@router.delete("/reports/{session_id}/share")
def unshare(session_id: str, user: User = Depends(current_user), db: Session = Depends(get_db)) -> dict:
    s = _own(db, user, session_id)
    s.share_token_hash = s.share_expires_at = None
    db.commit()
    return {"shared": False}


def shared_view(report: dict) -> dict:
    """What a link recipient sees: feedback and summary, without the raw transcript or appendix."""
    keep = {k: v for k, v in report.items() if k not in ("transcript", "appendix", "integrity")}
    keep["answers"] = [
        {k: v for k, v in a.items() if k not in ("voice", "lipsync")} for a in report["answers"]
    ]
    return keep


@router.get("/shared/{token}", dependencies=[Depends(share_limit)])
def get_shared(token: str, db: Session = Depends(get_db)) -> dict:
    s = db.query(InterviewSession).filter_by(share_token_hash=token_hash(token)).first()
    exp = s.share_expires_at if s and s.share_expires_at else None
    if (
        s is None
        or s.report is None
        or exp is None
        or (exp if exp.tzinfo else exp.replace(tzinfo=UTC)) < datetime.now(UTC)
    ):
        raise HTTPException(404, "link not found or expired")
    return shared_view(s.report)
