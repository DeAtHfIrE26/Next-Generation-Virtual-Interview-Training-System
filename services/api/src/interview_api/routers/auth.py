"""Accounts: register, login, logout, me."""

from __future__ import annotations

import re

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from interview_api.db import get_db
from interview_api.models import AuditLog, User
from interview_api.ratelimit import limiter
from interview_api.security import current_user, end_session, hash_password, start_session, verify_password
from interview_api.settings import get_settings

router = APIRouter(prefix="/auth", tags=["auth"])
auth_limit = limiter("auth", capacity=10, per_seconds=60)
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


class Credentials(BaseModel):
    email: str = Field(max_length=320)
    password: str = Field(min_length=10, max_length=200)


class Registration(Credentials):
    name: str = Field(default="", max_length=200)
    accept_terms: bool


def user_out(u: User) -> dict:
    return {"id": u.id, "email": u.email, "name": u.name, "plan": u.plan, "role": u.role}


@router.post("/register", status_code=201, dependencies=[Depends(auth_limit)])
def register(body: Registration, response: Response, db: Session = Depends(get_db)) -> dict:
    email = body.email.strip().lower()
    if not EMAIL_RE.match(email):
        raise HTTPException(422, "enter a valid email address")
    if not body.accept_terms:
        raise HTTPException(422, "you must accept the terms and privacy notice")
    if db.query(User).filter_by(email=email).first():
        raise HTTPException(409, "an account with this email already exists")
    role = "admin" if email in get_settings().admin_emails else "user"
    u = User(email=email, password_hash=hash_password(body.password), name=body.name.strip(), role=role)
    db.add(u)
    db.add(AuditLog(actor_id=u.id, action="register"))
    db.commit()
    start_session(db, u, response)
    return user_out(u)


@router.post("/login", dependencies=[Depends(auth_limit)])
def login(body: Credentials, response: Response, db: Session = Depends(get_db)) -> dict:
    u = db.query(User).filter_by(email=body.email.strip().lower()).first()
    if not verify_password(body.password, u.password_hash if u else None):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "email or password is incorrect")
    start_session(db, u, response)
    return user_out(u)


@router.post("/logout")
def logout(request: Request, response: Response, db: Session = Depends(get_db)) -> dict:
    end_session(db, request, response)
    return {"ok": True}


@router.get("/me")
def me(user: User = Depends(current_user)) -> dict:
    return user_out(user)
