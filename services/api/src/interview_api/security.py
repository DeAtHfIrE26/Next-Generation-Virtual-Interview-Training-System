"""Password hashing, cookie sessions and the current-user dependency."""

from __future__ import annotations

import hashlib
import secrets
from datetime import UTC, datetime, timedelta

from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerifyMismatchError
from fastapi import Depends, HTTPException, Request, Response, status
from sqlalchemy.orm import Session

from interview_api.db import get_db
from interview_api.models import AuthSession, User
from interview_api.settings import get_settings

COOKIE = "ic_session"
CSRF_HEADER = "x-ic-csrf"
_hasher = PasswordHasher()
# A fixed hash so unknown-email logins spend the same time as wrong-password logins.
_DUMMY_HASH = _hasher.hash("timing-equaliser-not-a-real-password")


def hash_password(pw: str) -> str:
    return _hasher.hash(pw)


def verify_password(pw: str, hashed: str | None) -> bool:
    try:
        return _hasher.verify(hashed or _DUMMY_HASH, pw) and hashed is not None
    except (VerifyMismatchError, InvalidHashError):
        return False


def token_hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def start_session(db: Session, user: User, response: Response) -> None:
    s = get_settings()
    token = secrets.token_urlsafe(32)
    expires = datetime.now(UTC) + timedelta(days=s.session_days)
    db.add(AuthSession(token_hash=token_hash(token), user_id=user.id, expires_at=expires))
    db.commit()
    response.set_cookie(
        COOKIE,
        token,
        httponly=True,
        secure=s.cookie_secure,
        samesite="lax",
        max_age=s.session_days * 86400,
        path="/",
    )


def end_session(db: Session, request: Request, response: Response) -> None:
    token = request.cookies.get(COOKIE)
    if token:
        db.query(AuthSession).filter_by(token_hash=token_hash(token)).delete()
        db.commit()
    response.delete_cookie(COOKIE, path="/")


def _aware(dt: datetime) -> datetime:
    return dt if dt.tzinfo else dt.replace(tzinfo=UTC)


def current_user(request: Request, db: Session = Depends(get_db)) -> User:
    token = request.cookies.get(COOKIE)
    if not token:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "sign in required")
    sess = db.get(AuthSession, token_hash(token))
    if sess is None or _aware(sess.expires_at) < datetime.now(UTC):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "session expired")
    user = db.get(User, sess.user_id)
    if user is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "account not found")
    return user


def admin_user(user: User = Depends(current_user)) -> User:
    if user.role != "admin":
        raise HTTPException(status.HTTP_403_FORBIDDEN, "admin only")
    return user
