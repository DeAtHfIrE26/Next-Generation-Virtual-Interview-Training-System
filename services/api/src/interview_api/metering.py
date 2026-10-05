"""Usage metering, per-session cost, plan limits and hard cost caps.

Costs are integers in micro-USD. Prices come from ``PRICE_TABLE_JSON`` (unit -> provider/model
-> micro-USD per unit) merged over the defaults below. Defaults contain only prices we have a
source for (Anthropic list prices); anything else costs 0 until configured, and the admin view
says so instead of showing a made-up number.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import cache

from interview_core.nlp.structured import CallRecord
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from interview_api.models import InterviewSession, LLMCall, UsageEvent, User

# micro-USD per unit. Anthropic list prices (USD per 1M tokens: Opus 5.5 4/20, Sonnet 5.5 2/10,
# Haiku 4.5 1/5) -> micro-USD per token = USD per 1M tokens.
DEFAULT_PRICES: dict[str, dict[str, float]] = {
    "llm_input_tokens": {
        "anthropic:claude-opus-5-5": 4.0,
        "anthropic:claude-sonnet-5-5": 2.0,
        "anthropic:claude-haiku-4-5": 1.0,
    },
    "llm_output_tokens": {
        "anthropic:claude-opus-5-5": 20.0,
        "anthropic:claude-sonnet-5-5": 10.0,
        "anthropic:claude-haiku-4-5": 5.0,
    },
    "asr_seconds": {},
    "tts_characters": {},
    "gpu_seconds": {},
    "code_runs": {},
}


@dataclass(frozen=True)
class PlanLimits:
    sessions_per_month: int
    session_cap_micro_usd: int
    server_tts: bool


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    return int(raw) if raw else default


def plan_limits(plan: str) -> PlanLimits:
    if plan in ("pro", "team"):
        return PlanLimits(
            _env_int("PRO_SESSIONS_PER_MONTH", 60), _env_int("PRO_SESSION_CAP_MICRO_USD", 2_000_000), True
        )
    return PlanLimits(
        _env_int("FREE_SESSIONS_PER_MONTH", 3), _env_int("FREE_SESSION_CAP_MICRO_USD", 250_000), False
    )


@cache
def prices() -> dict[str, dict[str, float]]:
    table = {k: dict(v) for k, v in DEFAULT_PRICES.items()}
    raw = os.getenv("PRICE_TABLE_JSON", "").strip()
    if raw:
        for unit, entries in json.loads(raw).items():
            table.setdefault(unit, {}).update(entries)
    return table


def unit_price(unit: str, provider_key: str) -> float | None:
    return prices().get(unit, {}).get(provider_key)


def record(db: Session, user: User, session_id: str | None, kind: str, provider_key: str, qty: float) -> int:
    price = unit_price(kind, provider_key)
    cost = round(qty * price) if price is not None else 0
    db.add(
        UsageEvent(
            user_id=user.id,
            session_id=session_id,
            kind=kind,
            provider=provider_key,
            quantity=qty,
            cost_micro_usd=cost,
        )
    )
    return cost


def record_llm_calls(db: Session, user: User, session_id: str | None, calls: list[CallRecord]) -> None:
    from interview_api.observability import record_model_call

    for c in calls:
        record_model_call(c.provider, c.task, c.raw_valid, c.used_fallback, c.latency_ms / 1000)
        db.add(
            LLMCall(
                session_id=session_id,
                task=c.task,
                provider=c.provider,
                model=c.model,
                raw_valid=c.raw_valid,
                delivered_valid=c.delivered_valid,
                used_fallback=c.used_fallback,
                attempts=c.attempts,
                latency_ms=c.latency_ms,
            )
        )
        if c.provider != "none":
            key = f"{c.provider}:{c.model}"
            record(db, user, session_id, "llm_input_tokens", key, c.input_tokens)
            record(db, user, session_id, "llm_output_tokens", key, c.output_tokens)


def session_cost(db: Session, session_id: str) -> int:
    return int(
        db.scalar(
            select(func.coalesce(func.sum(UsageEvent.cost_micro_usd), 0)).where(
                UsageEvent.session_id == session_id
            )
        )
        or 0
    )


def session_cap(user: User, sess: InterviewSession) -> int:
    return sess.cost_cap_micro_usd or plan_limits(user.plan).session_cap_micro_usd


def over_cap(db: Session, user: User, sess: InterviewSession) -> bool:
    return session_cost(db, sess.id) >= session_cap(user, sess)


def sessions_this_month(db: Session, user: User) -> int:
    start = datetime.now(UTC).replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    return int(
        db.scalar(
            select(func.count())
            .select_from(InterviewSession)
            .where(InterviewSession.user_id == user.id, InterviewSession.created_at >= start)
        )
        or 0
    )


def prices_configured() -> dict[str, bool]:
    return {unit: bool(entries) for unit, entries in prices().items()}
