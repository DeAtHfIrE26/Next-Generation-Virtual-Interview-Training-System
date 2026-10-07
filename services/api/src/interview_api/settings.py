"""Runtime configuration from environment variables (see .env.example)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import cache


def _bool(name: str, default: bool = False) -> bool:
    return os.getenv(name, str(default)).strip().lower() in ("1", "true", "yes", "on")


def _int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    return int(raw) if raw else default


@dataclass(frozen=True)
class Settings:
    env: str = field(default_factory=lambda: os.getenv("APP_ENV", "development"))
    database_url: str = field(default_factory=lambda: os.getenv("DATABASE_URL", "sqlite:///./var/dev.db"))
    app_base_url: str = field(default_factory=lambda: os.getenv("APP_BASE_URL", "http://localhost:3000"))
    cookie_secure: bool = field(
        default_factory=lambda: _bool("COOKIE_SECURE", os.getenv("APP_ENV") == "production")
    )
    session_days: int = field(default_factory=lambda: _int("AUTH_SESSION_DAYS", 14))
    biometric_retention_days: int = field(default_factory=lambda: _int("BIOMETRIC_RETENTION_DAYS", 30))
    raw_media_retention_days: int = field(default_factory=lambda: _int("RAW_MEDIA_RETENTION_DAYS", 0))
    session_retention_days: int = field(default_factory=lambda: _int("SESSION_RETENTION_DAYS", 365))
    feature_b2b_hiring: bool = field(default_factory=lambda: _bool("FEATURE_B2B_HIRING"))
    feature_neural_avatar: bool = field(default_factory=lambda: _bool("FEATURE_NEURAL_AVATAR"))
    neural_avatar_url: str = field(default_factory=lambda: os.getenv("NEURAL_AVATAR_URL", ""))
    max_body_bytes: int = field(default_factory=lambda: _int("MAX_BODY_BYTES", 8 * 1024 * 1024))
    admin_emails: tuple[str, ...] = field(
        default_factory=lambda: tuple(
            e.strip().lower() for e in os.getenv("ADMIN_EMAILS", "").split(",") if e.strip()
        )
    )

    @property
    def production(self) -> bool:
        return self.env == "production"


@cache
def get_settings() -> Settings:
    return Settings()
