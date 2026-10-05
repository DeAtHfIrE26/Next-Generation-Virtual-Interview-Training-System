import json
from pathlib import Path

import pytest
from conftest import signup
from helpers import speech_and_mouth, wav_b64
from interview_core.nlp.providers.base import LLMResponse

REPO = Path(__file__).resolve().parents[3]
ANSWER = (
    "At my previous job our nightly pipeline took six hours. I profiled each step, I rewrote the join "
    "and added an index. As a result the run dropped to 40 minutes."
)


class ScriptedProvider:
    name, model = "scripted", "scripted-1"

    def __init__(self, *items):
        self.items = list(items)
        self.calls = 0

    def complete_json(self, system, user, schema, *, timeout_s):
        self.calls += 1
        item = self.items.pop(0) if self.items else "not json"
        return LLMResponse(json.dumps(item), self.model, 1000, 500)


# ------------------------------------------------------------------ auth


def test_register_login_logout_and_csrf(client):
    u = signup(client)
    assert client.get("/auth/me").json()["email"] == "user@example.com"
    assert client.post("/auth/logout").status_code == 200
    assert client.get("/auth/me").status_code == 401
    bad = client.post("/auth/login", json={"email": "user@example.com", "password": "wrong password!!"})
    assert bad.status_code == 401
    ok = client.post("/auth/login", json={"email": "USER@example.com", "password": "correct horse battery"})
    assert ok.status_code == 200 and ok.json()["id"] == u["id"]
    nocsrf = client.post("/auth/logout", headers={"x-ic-csrf": ""})
    assert nocsrf.status_code == 403
    dup = client.post(
        "/auth/register",
        json={
            "email": "user@example.com",
            "password": "another password",
            "accept_terms": True,
            "age_confirmed": True,
        },
    )
    assert dup.status_code == 409


def test_registration_requires_adult_confirmation(client):
    r = client.post(
        "/auth/register",
        json={"email": "teen@example.com", "password": "correct horse battery", "accept_terms": True},
    )
    assert r.status_code == 422 and "18" in r.json()["detail"]


def test_login_is_rate_limited(client):
    signup(client)
    codes = [
        client.post(
            "/auth/login", json={"email": "x@example.com", "password": "wrong password!!"}
        ).status_code
        for _ in range(25)
    ]
    assert 429 in codes


def test_spoofed_forwarded_for_does_not_bypass_limits(client, monkeypatch):
    monkeypatch.delenv("TRUSTED_PROXY_HOPS", raising=False)
    codes = [
        client.post(
            "/auth/login",
            json={"email": "x@example.com", "password": "wrong password!!"},
            headers={"x-forwarded-for": f"10.0.0.{i}"},
        ).status_code
        for i in range(25)
    ]
    assert 429 in codes


def test_trusted_proxy_hops_uses_client_entry():
    from types import SimpleNamespace

    from interview_api.ratelimit import client_ip

    req = SimpleNamespace(
        headers={"x-forwarded-for": "6.6.6.6, 1.2.3.4, 76.76.21.1"}, client=SimpleNamespace(host="10.0.0.1")
    )
    import os

    os.environ["TRUSTED_PROXY_HOPS"] = "2"
    try:
        assert client_ip(req) == "1.2.3.4"
    finally:
        del os.environ["TRUSTED_PROXY_HOPS"]
    assert client_ip(req) == "10.0.0.1"


def test_admin_role_from_allowlist(client):
    assert signup(client, "admin@example.com")["role"] == "admin"


# ------------------------------------------------------------------ sessions


def _create(client, role="Backend Software Engineer", **form):
    data = {"role": role, "seniority": "mid", "length": "4", **form}
    pdf = (REPO / "legacy/desktop/sample_resume.pdf").read_bytes()  # synthetic fixture
    r = client.post("/sessions", data=data, files={"resume": ("cv.pdf", pdf, "application/pdf")})
    assert r.status_code == 201, r.text
    return r.json()


def test_session_requires_consent(client):
    signup(client, consents=())
    r = client.post("/sessions", data={"role": "Engineer"})
    assert r.status_code == 403 and "data_processing" in r.json()["detail"]


def test_full_offline_session_with_signals_and_report(client, user):
    created = _create(client)
    sid = created["id"]
    assert "Python" in " ".join(created["resume_profile"]["skills"]) or created["resume_profile"]["sections"]
    assert created["capabilities"]["llm"] is None  # offline: deterministic bank
    answered = 0
    while True:
        q = client.post(f"/sessions/{sid}/next").json()
        if q.get("done"):
            break
        assert "question" in q and "expected_points" not in q
        audio, times, values = speech_and_mouth(seed=answered)
        words = [{"word": w, "start": i * 0.4, "end": i * 0.4 + 0.3} for i, w in enumerate(ANSWER.split())]
        r = client.post(
            f"/sessions/{sid}/answer",
            json={
                "transcript": ANSWER,
                "words": words,
                "audio_wav": wav_b64(audio),
                "mouth": {"times": times, "values": values},
                "gaze_samples": [[i * 0.1, i % 10 != 0] for i in range(60)],
            },
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["evaluation"]["label"] == "experimental"
        assert body["signals"]["lipsync"]["decision"] == "match"
        assert body["signals"]["delivery"]["words_per_minute"] > 0
        assert body["signals"]["voice"]["status"] == "not_enrolled"
        answered += 1
    assert answered >= 4
    report = client.post(f"/sessions/{sid}/finish").json()
    assert report["summary"]["answers"] == answered and report["summary"]["label"] == "experimental"
    assert report["appendix"]["prototype_nine_factor"]["score"] >= 0
    assert all(a["lipsync"]["measured"] for a in report["answers"])
    assert client.get("/sessions").json()[0]["overall"] == report["summary"]["overall"]
    assert client.post(f"/sessions/{sid}/next").status_code == 409

    link = client.post(f"/reports/{sid}/share", json={"days": 7}).json()["url"]
    token = link.rsplit("/", 1)[1]
    shared = client.get(f"/shared/{token}").json()
    assert "transcript" not in shared and "appendix" not in shared
    client.delete(f"/reports/{sid}/share")
    assert client.get(f"/shared/{token}").status_code == 404


def test_lipsync_mismatch_raises_integrity_notice(client, user):
    sid = _create(client)["id"]
    client.post(f"/sessions/{sid}/next")
    audio, times, values = speech_and_mouth(seed=3, mismatch=True)
    r = client.post(
        f"/sessions/{sid}/answer",
        json={"transcript": ANSWER, "audio_wav": wav_b64(audio), "mouth": {"times": times, "values": values}},
    ).json()
    assert r["signals"]["lipsync"]["decision"] == "mismatch"
    assert r["notices"][0]["event"] == "lipsync_mismatch" and not r["notices"][0]["end_session"]


def test_answer_without_question_and_double_answer(client, user):
    sid = _create(client)["id"]
    assert client.post(f"/sessions/{sid}/answer", json={"transcript": "hi"}).status_code == 409
    client.post(f"/sessions/{sid}/next")
    assert client.post(f"/sessions/{sid}/answer", json={"transcript": "hi"}).status_code == 200
    assert client.post(f"/sessions/{sid}/answer", json={"transcript": "hi"}).status_code == 409


def test_sessions_are_private(client, user):
    sid = _create(client)["id"]
    client.post("/auth/logout")
    signup(client, "other@example.com")
    assert client.get(f"/sessions/{sid}").status_code == 404
    assert client.post(f"/sessions/{sid}/next").status_code == 404


def test_phone_debounce_and_proctored_policy(client, user):
    sid = _create(client, mode="proctored")["id"]
    obs = [{"t": i * 0.5, "type": "phone", "present": True} for i in range(3)]
    r = client.post(f"/sessions/{sid}/events", json={"observations": obs[:2]}).json()
    assert r["notices"] == []
    r = client.post(f"/sessions/{sid}/events", json={"observations": obs[2:]}).json()
    assert r["notices"][0]["event"] == "phone"
    for k in range(2):  # two more phone episodes end a proctored session (limit 3)
        client.post(
            f"/sessions/{sid}/events",
            json={"observations": [{"t": 10 + k, "type": "phone", "present": False}]},
        )
        r = client.post(
            f"/sessions/{sid}/events",
            json={
                "observations": [
                    {"t": 10.1 + k + i * 0.1, "type": "phone", "present": True} for i in range(3)
                ]
            },
        ).json()
    assert r["status"] == "ended_by_policy"
    assert client.post(f"/sessions/{sid}/next").status_code == 409


def test_coaching_mode_never_ends_on_integrity(client, user):
    sid = _create(client)["id"]
    for k in range(6):
        client.post(
            f"/sessions/{sid}/events",
            json={
                "observations": [
                    {"t": k * 10 + i * 0.1, "type": "second_person", "present": True} for i in range(3)
                ]
                + [{"t": k * 10 + 1, "type": "second_person", "present": False}]
            },
        )
    assert client.get(f"/sessions/{sid}").json()["status"] == "active"


def test_coding_challenge_attached_and_graded(client, user):
    sid = _create(client, role="Data Analyst")["id"]
    challenge = None
    for _ in range(6):
        q = client.post(f"/sessions/{sid}/next").json()
        if q.get("done"):
            break
        challenge = challenge or q.get("challenge")
        client.post(f"/sessions/{sid}/answer", json={"transcript": ANSWER})
    assert challenge is not None and "tests" not in challenge
    if challenge["languages"] == ["sql"]:
        r = client.post(
            f"/sessions/{sid}/code", json={"language": "sql", "source": "SELECT 1", "final": True}
        ).json()
        assert r["total"] >= 1 and r["passed"] == 0
    else:
        r = client.post(f"/sessions/{sid}/code", json={"language": "python", "source": "print(1)"}).json()
        assert "not configured" in r["error"]


# ------------------------------------------------------------------ limits and cost caps


def test_plan_session_limit(client, user, monkeypatch):
    monkeypatch.setenv("FREE_SESSIONS_PER_MONTH", "2")
    _create(client)
    _create(client)
    r = client.post("/sessions", data={"role": "Engineer"})
    assert r.status_code == 402


def test_cost_cap_switches_to_offline_bank(client, user, monkeypatch):
    from interview_api import metering, runtime

    q = {
        "question": "Tell me about a hard bug you fixed.",
        "category": "behavioral",
        "difficulty": 3,
        "competency": "debugging",
        "rationale": "r",
        "expected_points": ["a", "b"],
    }
    prov = ScriptedProvider(q)
    monkeypatch.setattr(runtime, "llm_provider", lambda: prov)
    monkeypatch.setenv(
        "PRICE_TABLE_JSON",
        json.dumps(
            {
                "llm_input_tokens": {"scripted:scripted-1": 1.0},
                "llm_output_tokens": {"scripted:scripted-1": 1.0},
            }
        ),
    )
    monkeypatch.setenv("FREE_SESSION_CAP_MICRO_USD", "1000")
    metering.prices.cache_clear()
    sid = _create(client)["id"]
    first = client.post(f"/sessions/{sid}/next").json()
    assert first["source"] == "llm" and prov.calls == 1
    client.post(f"/sessions/{sid}/answer", json={"transcript": ANSWER})
    second = client.post(f"/sessions/{sid}/next").json()
    assert second["source"] in ("bank", "follow_up") and prov.calls == 1  # cap reached: no more paid calls


# ------------------------------------------------------------------ privacy


def test_export_and_delete_account(client, user):
    sid = _create(client)["id"]
    client.post(f"/sessions/{sid}/next")
    exp = client.get("/privacy/export").json()
    assert exp["account"]["email"] == "user@example.com" and exp["sessions"][0]["id"] == sid
    assert any(c["kind"] == "data_processing" for c in exp["consents"])
    assert client.delete("/privacy/account").json()["deleted"]
    assert client.get("/auth/me").status_code == 401
    r = client.post("/auth/login", json={"email": "user@example.com", "password": "correct horse battery"})
    assert r.status_code == 401


def test_retention_purge(client, user):
    from datetime import UTC, datetime, timedelta

    from interview_api.db import get_db
    from interview_api.routers.privacy import purge_expired

    client.post("/sessions", data={"role": "Engineer"})
    db = next(get_db())
    out = purge_expired(db, now=datetime.now(UTC) + timedelta(days=400))
    assert out["sessions"] == 1 and out["auth_sessions"] >= 1


@pytest.mark.parametrize("payload", [{"transcript": "x" * 9000}, {"audio_wav": "!!notbase64!!"}])
def test_input_validation(client, user, payload):
    sid = _create(client)["id"]
    client.post(f"/sessions/{sid}/next")
    assert client.post(f"/sessions/{sid}/answer", json=payload).status_code == 422
