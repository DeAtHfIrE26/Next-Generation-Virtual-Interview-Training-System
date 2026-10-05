import hashlib
import hmac
import json
import time

import httpx
import pytest
from conftest import signup

STRIPE_SECRET = "whsec_test_secret_value"
RZP_SECRET = "rzp_webhook_secret_value"


@pytest.fixture
def billing_env(monkeypatch):
    monkeypatch.setenv("STRIPE_SECRET_KEY", "sk_test_dummy_not_real")
    monkeypatch.setenv("STRIPE_PRICE_PRO", "price_123")
    monkeypatch.setenv("STRIPE_WEBHOOK_SECRET", STRIPE_SECRET)
    monkeypatch.setenv("RAZORPAY_KEY_ID", "rzp_test_dummy")
    monkeypatch.setenv("RAZORPAY_KEY_SECRET", "secret")
    monkeypatch.setenv("RAZORPAY_PLAN_PRO", "plan_123")
    monkeypatch.setenv("RAZORPAY_WEBHOOK_SECRET", RZP_SECRET)


def stripe_post(client, event, secret=STRIPE_SECRET, ts=None):
    payload = json.dumps(event).encode()
    ts = ts or int(time.time())
    sig = hmac.new(secret.encode(), f"{ts}.".encode() + payload, hashlib.sha256).hexdigest()
    return client.post(
        "/billing/webhooks/stripe",
        content=payload,
        headers={"stripe-signature": f"t={ts},v1={sig}", "content-type": "application/json"},
    )


def test_stripe_checkout_and_webhook_upgrade_and_cancel(client, billing_env, monkeypatch):
    user = signup(client)
    captured = {}

    def fake_post(url, **kw):
        captured.update(url=url, data=kw.get("data"))
        return httpx.Response(200, json={"url": "https://checkout.stripe.test/s/1"})

    monkeypatch.setattr(httpx, "post", fake_post)
    assert client.post("/billing/stripe/checkout", json={"plan": "pro"}).json()["url"].startswith("https://")
    assert captured["data"]["client_reference_id"] == user["id"]
    assert client.get("/auth/me").json()["plan"] == "free"  # redirect alone never upgrades

    done = {
        "id": "evt_1",
        "type": "checkout.session.completed",
        "data": {
            "object": {
                "subscription": "sub_1",
                "client_reference_id": user["id"],
                "metadata": {"plan": "pro"},
            }
        },
    }
    assert stripe_post(client, done).json() == {"ok": True}
    assert client.get("/auth/me").json()["plan"] == "pro"
    assert stripe_post(client, done).json() == {"duplicate": True}

    cancel = {
        "id": "evt_2",
        "type": "customer.subscription.deleted",
        "data": {"object": {"id": "sub_1", "status": "canceled", "metadata": {"user_id": user["id"]}}},
    }
    stripe_post(client, cancel)
    assert client.get("/auth/me").json()["plan"] == "free"


def test_stripe_rejects_bad_and_stale_signatures(client, billing_env):
    signup(client)
    ev = {"id": "evt_x", "type": "checkout.session.completed", "data": {"object": {}}}
    assert stripe_post(client, ev, secret="wrong").status_code == 400
    assert stripe_post(client, ev, ts=int(time.time()) - 3600).status_code == 400


def test_razorpay_subscription_webhook(client, billing_env, monkeypatch):
    user = signup(client)
    monkeypatch.setattr(
        httpx,
        "post",
        lambda url, **kw: httpx.Response(200, json={"id": "sub_R1", "short_url": "https://rzp.io/x"}),
    )
    assert (
        client.post("/billing/razorpay/subscription", json={"plan": "pro"}).json()["subscription_id"]
        == "sub_R1"
    )
    event = {
        "event": "subscription.activated",
        "payload": {
            "subscription": {
                "entity": {
                    "id": "sub_R1",
                    "notes": {"user_id": user["id"], "plan": "pro"},
                    "current_end": int(time.time()) + 86400,
                }
            }
        },
    }
    payload = json.dumps(event).encode()
    sig = hmac.new(RZP_SECRET.encode(), payload, hashlib.sha256).hexdigest()
    bad = client.post("/billing/webhooks/razorpay", content=payload, headers={"x-razorpay-signature": "00"})
    assert bad.status_code == 400
    ok = client.post(
        "/billing/webhooks/razorpay",
        content=payload,
        headers={"x-razorpay-signature": sig, "x-razorpay-event-id": "e1"},
    )
    assert ok.json() == {"ok": True} and client.get("/auth/me").json()["plan"] == "pro"
    st = client.get("/billing/status").json()
    assert st["subscription"]["provider"] == "razorpay" and st["sessions_per_month"] == 60


def test_billing_not_configured_is_503(client):
    signup(client)
    assert client.post("/billing/stripe/checkout", json={"plan": "pro"}).status_code == 503


def test_admin_requires_admin_and_reports_costs(client, monkeypatch):
    signup(client)
    assert client.get("/admin/overview").status_code == 403
    client.post("/auth/logout")
    signup(client, "admin@example.com")
    sid = client.post("/sessions", data={"role": "Data Analyst"}).json()["id"]
    client.post(f"/sessions/{sid}/next")
    client.post(f"/sessions/{sid}/answer", json={"transcript": "I built a dashboard."})
    client.post("/metrics/latency", json={"metric": "question_to_first_avatar_frame", "ms": 640})
    ov = client.get("/admin/overview").json()
    assert ov["llm_quality"]["none/question"]["fallback_rate"] == 1.0
    assert ov["latency"]["question_to_first_avatar_frame"]["p95_ms"] == 640
    assert ov["cost"]["prices_configured"]["llm_input_tokens"] is True
    manifest = client.get("/admin/export/llm_schema_validity").text.strip().splitlines()
    assert json.loads(manifest[0])["consent_id"] == "system"


def test_b2b_seats(client):
    signup(client, "member1@example.com")
    client.post("/auth/logout")
    signup(client, "member2@example.com")
    client.post("/auth/logout")
    signup(client, "admin@example.com")
    org = client.post("/admin/orgs", json={"name": "Placement Cell", "seats": 1}).json()
    assert (
        client.post(f"/admin/orgs/{org['id']}/members", json={"email": "member1@example.com"}).json()[
            "seats_used"
        ]
        == 1
    )
    assert (
        client.post(f"/admin/orgs/{org['id']}/members", json={"email": "member2@example.com"}).status_code
        == 409
    )
