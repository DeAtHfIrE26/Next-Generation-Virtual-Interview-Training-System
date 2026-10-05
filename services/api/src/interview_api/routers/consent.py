"""Consent records (DPDP: specific, informed, revocable; latest record wins)."""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from interview_api.db import get_db
from interview_api.models import BiometricTemplate, Consent, User
from interview_api.security import current_user

router = APIRouter(prefix="/consent", tags=["consent"])

CONSENT_VERSION = "2026-10-v1"
Kind = Literal["data_processing", "biometric_face", "biometric_voice", "store_recordings", "model_training"]
TEXT: dict[str, str] = {
    "data_processing": "Process my resume, answers and session data to run practice interviews and give me feedback.",
    "biometric_face": "Create an encrypted face template from my camera to check that the same person stays in "
    "the session. Raw images are not stored.",
    "biometric_voice": "Create an encrypted voice template from short phrases I read, to check the same person "
    "is answering. Raw recordings are not stored.",
    "store_recordings": "Keep audio/video recordings of my sessions so I can replay them (optional).",
    "model_training": "Use my de-identified session data to improve the models (optional).",
}


class ConsentIn(BaseModel):
    kind: Kind
    granted: bool


def latest(db: Session, user_id: str) -> dict[str, bool]:
    out: dict[str, bool] = dict.fromkeys(TEXT, False)
    for c in db.query(Consent).filter_by(user_id=user_id).order_by(Consent.created_at):
        out[c.kind] = c.granted
    return out


def require(db: Session, user: User, kind: str) -> None:
    if not latest(db, user.id).get(kind):
        raise HTTPException(403, f"consent '{kind}' is required for this action")


@router.get("")
def get_consents(user: User = Depends(current_user), db: Session = Depends(get_db)) -> dict:
    return {"version": CONSENT_VERSION, "text": TEXT, "granted": latest(db, user.id)}


@router.post("")
def set_consent(body: ConsentIn, user: User = Depends(current_user), db: Session = Depends(get_db)) -> dict:
    db.add(Consent(user_id=user.id, kind=body.kind, granted=body.granted, version=CONSENT_VERSION))
    # Withdrawing biometric consent deletes the corresponding template immediately.
    if not body.granted and body.kind in ("biometric_face", "biometric_voice"):
        db.query(BiometricTemplate).filter_by(user_id=user.id, kind=body.kind.split("_")[1]).delete()
    db.commit()
    return {"granted": latest(db, user.id)}
