import json
import logging

from conftest import signup
from interview_api.observability import JsonFormatter, redact


def test_redaction_masks_credentials_and_personal_data():
    fake_key = (
        "sk-" + "ant-" + "abcdefghijklmnopqrstu"
    )  # built at runtime so the repo secret scanner stays quiet
    s = redact(
        f"email priya@example.com phone +91 98765 43210 key {fake_key} token=abc123 Authorization: Bearer xyz"
    )
    assert (
        "priya@" not in s
        and "98765" not in s
        and "sk-ant-abc" not in s
        and "abc123" not in s
        and "xyz" not in s
    )


def test_json_log_lines_are_redacted():
    rec = logging.LogRecord("t", logging.INFO, __file__, 1, "user %s signed in", ("a.b@example.com",), None)
    line = json.loads(JsonFormatter().format(rec))
    assert line["msg"] == "user <REDACTED:email> signed in" and line["level"] == "INFO"


def test_metrics_and_request_id(client):
    signup(client)
    r = client.get("/auth/me", headers={"x-request-id": "req-123"})
    assert r.headers["x-request-id"] == "req-123"
    sid = client.post("/sessions", data={"role": "Engineer"}).json()["id"]
    client.post(f"/sessions/{sid}/turn", json={})
    m = client.get("/metrics").text
    assert 'http_requests_total{method="GET",route="/auth/me",status="200"}' in m
    assert 'model_calls_total{outcome="fallback",provider="emergency",task="interviewer"}' in m
    assert "priya" not in m


def test_metrics_token(client, monkeypatch):
    monkeypatch.setenv("METRICS_TOKEN", "t0ken")
    assert client.get("/metrics").status_code == 401
    assert client.get("/metrics", headers={"authorization": "Bearer t0ken"}).status_code == 200


def test_shared_secret_gate(client, monkeypatch):
    monkeypatch.setenv("API_SHARED_SECRET", "dummy-shared")
    assert client.get("/health").status_code == 200
    assert client.get("/auth/me").status_code == 404
    r = client.get("/auth/me", headers={"x-ic-internal": "dummy-shared"})
    assert r.status_code == 401
