"""Billing: Stripe (international) and Razorpay (India) subscriptions with verified webhooks.

Plan changes happen only from signed webhooks, never from the browser redirect, and each
webhook event id is processed once.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import time
from datetime import UTC, datetime
from typing import Literal

import httpx
from fastapi import APIRouter, Depends, Header, HTTPException, Request
from pydantic import BaseModel
from sqlalchemy.orm import Session

from interview_api import metering
from interview_api.db import get_db
from interview_api.models import AuditLog, Subscription, UsageEvent, User, WebhookEvent
from interview_api.security import current_user
from interview_api.settings import get_settings

router = APIRouter(prefix="/billing", tags=["billing"])
STRIPE_TOLERANCE_S = 300
ACTIVE = {"active", "trialing", "authenticated", "charged"}


def _env(name: str) -> str:
    v = os.getenv(name, "")
    if not v:
        raise HTTPException(503, f"billing is not configured ({name})")
    return v


@router.get("/status")
def status(user: User = Depends(current_user), db: Session = Depends(get_db)) -> dict:
    limits = metering.plan_limits(user.plan)
    sub = db.query(Subscription).filter_by(user_id=user.id).order_by(Subscription.updated_at.desc()).first()
    start = datetime.now(UTC).replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    used = sum(
        u.cost_micro_usd
        for u in db.query(UsageEvent).filter(UsageEvent.user_id == user.id, UsageEvent.created_at >= start)
    )
    return {
        "plan": user.plan,
        "sessions_this_month": metering.sessions_this_month(db, user),
        "sessions_per_month": limits.sessions_per_month,
        "session_cap_micro_usd": limits.session_cap_micro_usd,
        "usage_this_month_micro_usd": used,
        "subscription": None
        if sub is None
        else {"provider": sub.provider, "status": sub.status, "plan": sub.plan},
        "providers": {
            "stripe": bool(os.getenv("STRIPE_SECRET_KEY")),
            "razorpay": bool(os.getenv("RAZORPAY_KEY_ID")),
        },
    }


class CheckoutIn(BaseModel):
    plan: Literal["pro"] = "pro"


# ----------------------------------------------------------------------------- Stripe


@router.post("/stripe/checkout")
def stripe_checkout(body: CheckoutIn, user: User = Depends(current_user)) -> dict:
    key, price = _env("STRIPE_SECRET_KEY"), _env("STRIPE_PRICE_PRO")
    base = get_settings().app_base_url
    r = httpx.post(
        "https://api.stripe.com/v1/checkout/sessions",
        auth=(key, ""),
        timeout=15,
        data={
            "mode": "subscription",
            "line_items[0][price]": price,
            "line_items[0][quantity]": "1",
            "client_reference_id": user.id,
            "customer_email": user.email,
            "metadata[user_id]": user.id,
            "metadata[plan]": body.plan,
            "subscription_data[metadata][user_id]": user.id,
            "subscription_data[metadata][plan]": body.plan,
            "success_url": f"{base}/settings/billing?checkout=success",
            "cancel_url": f"{base}/settings/billing",
        },
    )
    if r.status_code >= 400:
        raise HTTPException(502, "payment provider error")
    return {"url": r.json()["url"]}


def verify_stripe_signature(payload: bytes, header: str, secret: str, now: float | None = None) -> None:
    parts = dict(p.split("=", 1) for p in header.split(",") if "=" in p)
    try:
        ts = int(parts.get("t", ""))
    except ValueError as e:
        raise HTTPException(400, "bad signature header") from e
    if abs((now or time.time()) - ts) > STRIPE_TOLERANCE_S:
        raise HTTPException(400, "stale webhook")
    expected = hmac.new(secret.encode(), f"{ts}.".encode() + payload, hashlib.sha256).hexdigest()
    sigs = [v for k, v in (p.split("=", 1) for p in header.split(",") if "=" in p) if k == "v1"]
    if not any(hmac.compare_digest(expected, s) for s in sigs):
        raise HTTPException(400, "invalid signature")


def _once(db: Session, provider: str, event_id: str) -> bool:
    if db.get(WebhookEvent, event_id):
        return False
    db.add(WebhookEvent(id=event_id, provider=provider))
    return True


def _apply(
    db: Session,
    provider: str,
    external_id: str,
    user_id: str | None,
    plan: str,
    status: str,
    period_end: int | None = None,
) -> None:
    sub = db.query(Subscription).filter_by(external_id=external_id).first()
    if sub is None:
        if not user_id:
            return
        sub = Subscription(
            user_id=user_id, provider=provider, external_id=external_id, plan=plan, status=status
        )
        db.add(sub)
    sub.status, sub.plan, sub.updated_at = status, plan or sub.plan, datetime.now(UTC)
    if period_end:
        sub.current_period_end = datetime.fromtimestamp(period_end, UTC)
    user = db.get(User, sub.user_id) if sub.user_id else None
    if user is not None and user.org_id is None:
        user.plan = sub.plan if status in ACTIVE else "free"
        db.add(AuditLog(actor_id=user.id, action=f"plan_{user.plan}", target=f"{provider}:{external_id}"))


@router.post("/webhooks/stripe")
async def stripe_webhook(
    request: Request, stripe_signature: str = Header(default=""), db: Session = Depends(get_db)
) -> dict:
    payload = await request.body()
    verify_stripe_signature(payload, stripe_signature, _env("STRIPE_WEBHOOK_SECRET"))
    event = json.loads(payload)
    if not _once(db, "stripe", event["id"]):
        return {"duplicate": True}
    obj = event["data"]["object"]
    kind = event["type"]
    if kind == "checkout.session.completed" and obj.get("subscription"):
        _apply(
            db,
            "stripe",
            obj["subscription"],
            obj.get("client_reference_id"),
            (obj.get("metadata") or {}).get("plan", "pro"),
            "active",
        )
    elif kind in ("customer.subscription.updated", "customer.subscription.deleted"):
        meta = obj.get("metadata") or {}
        status = "canceled" if kind.endswith("deleted") else obj.get("status", "active")
        _apply(
            db,
            "stripe",
            obj["id"],
            meta.get("user_id"),
            meta.get("plan", "pro"),
            status,
            obj.get("current_period_end"),
        )
    db.commit()
    return {"ok": True}


# ----------------------------------------------------------------------------- Razorpay


@router.post("/razorpay/subscription")
def razorpay_subscription(body: CheckoutIn, user: User = Depends(current_user)) -> dict:
    key_id, secret, plan_id = _env("RAZORPAY_KEY_ID"), _env("RAZORPAY_KEY_SECRET"), _env("RAZORPAY_PLAN_PRO")
    r = httpx.post(
        "https://api.razorpay.com/v1/subscriptions",
        auth=(key_id, secret),
        timeout=15,
        json={
            "plan_id": plan_id,
            "total_count": 12,
            "customer_notify": 1,
            "notes": {"user_id": user.id, "plan": body.plan},
        },
    )
    if r.status_code >= 400:
        raise HTTPException(502, "payment provider error")
    data = r.json()
    return {"subscription_id": data["id"], "url": data.get("short_url"), "key_id": key_id}


def verify_razorpay_signature(payload: bytes, signature: str, secret: str) -> None:
    expected = hmac.new(secret.encode(), payload, hashlib.sha256).hexdigest()
    if not signature or not hmac.compare_digest(expected, signature):
        raise HTTPException(400, "invalid signature")


RAZORPAY_STATUS = {
    "subscription.activated": "active",
    "subscription.charged": "active",
    "subscription.authenticated": "authenticated",
    "subscription.pending": "past_due",
    "subscription.halted": "halted",
    "subscription.cancelled": "canceled",
    "subscription.completed": "completed",
}


@router.post("/webhooks/razorpay")
async def razorpay_webhook(
    request: Request,
    x_razorpay_signature: str = Header(default=""),
    x_razorpay_event_id: str = Header(default=""),
    db: Session = Depends(get_db),
) -> dict:
    payload = await request.body()
    verify_razorpay_signature(payload, x_razorpay_signature, _env("RAZORPAY_WEBHOOK_SECRET"))
    event = json.loads(payload)
    event_id = x_razorpay_event_id or hashlib.sha256(payload).hexdigest()
    if not _once(db, "razorpay", event_id):
        return {"duplicate": True}
    status = RAZORPAY_STATUS.get(event.get("event", ""))
    sub = ((event.get("payload") or {}).get("subscription") or {}).get("entity") or {}
    if status and sub.get("id"):
        notes = sub.get("notes") or {}
        _apply(
            db,
            "razorpay",
            sub["id"],
            notes.get("user_id"),
            notes.get("plan", "pro"),
            status,
            sub.get("current_end"),
        )
    db.commit()
    return {"ok": True}
