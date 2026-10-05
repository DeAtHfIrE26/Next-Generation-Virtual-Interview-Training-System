"""E2/E3 enrolment: liveness challenge, face template, prompted-phrase voice template."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from interview_core.adapters import factory
from interview_core.crypto import encrypt_template
from interview_core.face import FaceVerifier
from interview_core.face.liveness import Challenge as LiveChallenge
from interview_core.face.liveness import issue_challenge, series_digest, verify_challenge
from interview_core.voice import PhraseChallenge, VoiceVerifier, check_phrase, issue_phrase
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from interview_api import runtime
from interview_api.db import get_db
from interview_api.media import decode_image, decode_wav
from interview_api.models import AuditLog, BiometricTemplate, Challenge, User
from interview_api.ratelimit import limiter
from interview_api.routers.consent import require
from interview_api.security import current_user
from interview_api.settings import get_settings

router = APIRouter(prefix="/enrollment", tags=["enrollment"])
enrol_limit = limiter("enrol", capacity=20, per_seconds=300)


def _aware(dt: datetime) -> datetime:
    return dt if dt.tzinfo else dt.replace(tzinfo=UTC)


def _store(db: Session, user: User, kind: str, template) -> None:
    db.query(BiometricTemplate).filter_by(user_id=user.id, kind=kind).delete()
    blob = encrypt_template(template, user.id, runtime.key_provider())
    days = max(get_settings().biometric_retention_days, 1)
    db.add(
        BiometricTemplate(
            user_id=user.id,
            kind=kind,
            model_id=template.model_id,
            blob=blob.to_json(),
            expires_at=datetime.now(UTC) + timedelta(days=days),
        )
    )
    db.add(AuditLog(actor_id=user.id, action=f"enrol_{kind}", target=template.model_id))


@router.get("/status")
def status(user: User = Depends(current_user), db: Session = Depends(get_db)) -> dict:
    now = datetime.now(UTC)
    t = {
        r.kind: _aware(r.expires_at).isoformat()
        for r in db.query(BiometricTemplate).filter_by(user_id=user.id)
        if _aware(r.expires_at) > now
    }
    return {"face": "face" in t, "voice": "voice" in t, "expires": t, "capabilities": runtime.capabilities()}


@router.post("/liveness/challenge", dependencies=[Depends(enrol_limit)])
def liveness_challenge(user: User = Depends(current_user), db: Session = Depends(get_db)) -> dict:
    require(db, user, "biometric_face")
    ch = issue_challenge()
    db.add(
        Challenge(
            nonce=ch.nonce, user_id=user.id, kind="liveness", payload=ch.to_dict(), expires_at=ch.expires_at
        )
    )
    db.commit()
    return ch.to_dict()


class FaceEnrolment(BaseModel):
    nonce: str
    series: dict[str, Any]
    faces: list[str] = Field(min_length=5, max_length=20)  # base64 face crops


def _consume_liveness(db: Session, user: User, nonce: str, series: dict[str, Any]):
    row = db.get(Challenge, nonce)
    if row is None or row.user_id != user.id or row.kind != "liveness":
        raise HTTPException(404, "unknown challenge")
    digest = series_digest(series)
    replayed = (
        db.query(Challenge).filter(Challenge.user_id == user.id, Challenge.series_digest == digest).first()
    )
    p = row.payload
    ch = LiveChallenge(
        p["nonce"],
        tuple(p["steps"]),
        datetime.fromisoformat(p["issued_at"]),
        datetime.fromisoformat(p["expires_at"]),
    )
    result = verify_challenge(ch, series, already_used=row.used or replayed is not None)
    row.used, row.series_digest = True, digest
    db.commit()
    return result


@router.post("/face", dependencies=[Depends(enrol_limit)])
def enrol_face(
    body: FaceEnrolment, user: User = Depends(current_user), db: Session = Depends(get_db)
) -> dict:
    require(db, user, "biometric_face")
    live = _consume_liveness(db, user, body.nonce, body.series)
    if not live.passed:
        raise HTTPException(422, {"message": "liveness check failed", "reasons": live.reasons})
    emb = runtime.face_embedder()
    if emb is None:
        raise HTTPException(503, "face verification is not configured on this server")
    faces = [decode_image(f) for f in body.faces]  # decoded in memory, discarded after embedding
    template = FaceVerifier(emb, factory.face_threshold()).enrol(faces)
    _store(db, user, "face", template)
    db.commit()
    return {"enrolled": True, "samples": template.n_samples, "liveness": live.detected}


@router.post("/voice/phrase", dependencies=[Depends(enrol_limit)])
def voice_phrase(user: User = Depends(current_user), db: Session = Depends(get_db)) -> dict:
    require(db, user, "biometric_voice")
    ch = issue_phrase()
    db.add(
        Challenge(
            nonce=ch.nonce,
            user_id=user.id,
            kind="phrase",
            payload={"phrase": ch.phrase, "expires_at": ch.expires_at.isoformat()},
            expires_at=ch.expires_at,
        )
    )
    db.commit()
    return {"nonce": ch.nonce, "phrase": ch.phrase, "expires_at": ch.expires_at.isoformat()}


class PhraseRecording(BaseModel):
    nonce: str
    audio_wav: str
    client_transcript: str = Field(default="", max_length=500)


class VoiceEnrolment(BaseModel):
    recordings: list[PhraseRecording] = Field(min_length=3, max_length=6)


@router.post("/voice", dependencies=[Depends(enrol_limit)])
def enrol_voice(
    body: VoiceEnrolment, user: User = Depends(current_user), db: Session = Depends(get_db)
) -> dict:
    require(db, user, "biometric_voice")
    emb = runtime.speaker_embedder()
    if emb is None:
        raise HTTPException(503, "voice verification is not configured on this server")
    asr = runtime.asr_provider()
    utterances, checks = [], []
    for rec in body.recordings:
        row = db.get(Challenge, rec.nonce)
        if row is None or row.user_id != user.id or row.kind != "phrase" or row.used:
            raise HTTPException(422, "phrase challenge missing, expired or already used")
        row.used = True
        audio, sr = decode_wav(rec.audio_wav)
        heard = asr.transcribe(audio, sr).text if asr else rec.client_transcript
        ch = PhraseChallenge(
            row.nonce, row.payload["phrase"], datetime.fromisoformat(row.payload["expires_at"])
        )
        ok, wer = check_phrase(ch, heard)
        checks.append(
            {"ok": ok, "wer": round(wer, 2), "checked_by": "server_asr" if asr else "client_reported"}
        )
        if not ok:
            db.commit()
            raise HTTPException(
                422,
                {"message": "the recording did not match the phrase; please read it again", "checks": checks},
            )
        utterances.append((audio, sr))
    try:
        template = VoiceVerifier(emb, factory.voice_threshold()).enrol(utterances)
    except ValueError as e:
        db.commit()
        raise HTTPException(422, str(e)) from e
    _store(db, user, "voice", template)
    db.commit()
    return {"enrolled": True, "phrases": checks}


@router.delete("")
def delete_enrolment(user: User = Depends(current_user), db: Session = Depends(get_db)) -> dict:
    n = db.query(BiometricTemplate).filter_by(user_id=user.id).delete()
    db.add(AuditLog(actor_id=user.id, action="delete_biometrics", target=str(n)))
    db.commit()
    return {"deleted": n}
