import json
from pathlib import Path

import pytest
from conftest import signup
from fake_llm import FakeInterviewerLLM
from helpers import speech_and_mouth, wav_b64

REPO = Path(__file__).resolve().parents[3]
ANSWER = (
    "At my previous job our nightly pipeline took six hours. I profiled each step, I rewrote the join "
    "and added an index. As a result the run dropped to 40 minutes."
)


@pytest.fixture
def llm(monkeypatch):
    from interview_api import runtime

    fake = FakeInterviewerLLM()
    monkeypatch.setattr(runtime, "llm_chain", lambda: (fake,))
    return fake


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
    data = {"role": role, "seniority": "mid", "duration_minutes": "10", "interview_type": "technical", **form}
    pdf = (REPO / "legacy/desktop/sample_resume.pdf").read_bytes()  # synthetic fixture
    r = client.post("/sessions", data=data, files={"resume": ("cv.pdf", pdf, "application/pdf")})
    assert r.status_code == 201, r.text
    return r.json()


def _answer(client, sid, **body):
    r = client.post(f"/sessions/{sid}/turn", json=body)
    assert r.status_code == 200, r.text
    return r.json()


def test_session_requires_consent(client):
    signup(client, consents=())
    r = client.post("/sessions", data={"role": "Engineer"})
    assert r.status_code == 403 and "data_processing" in r.json()["detail"]


def test_options_expose_parameters_and_personas(client, user):
    o = client.get("/sessions/options").json()
    assert "system_design" in o["interview_types"] and "final" in o["rounds"]
    assert {p["id"] for p in o["personas"]} >= {"maya", "daniel", "priya", "arjun"}


def test_full_session_with_signals_and_report(client, user, llm):
    created = _create(client, company="Acme", skills="Postgres, Kafka", round="onsite")
    sid = created["id"]
    assert created["capabilities"]["llm"] == "scripted"
    first = _answer(client, sid)["turn"]
    assert first["action"] == "open" and not first["emergency"]
    info = client.get(f"/sessions/{sid}").json()
    assert info["params"]["company"] == "Acme" and info["params"]["skills"] == ["Postgres", "Kafka"]
    assert [c["name"] for c in info["blueprint"]["competencies"]] == [
        "System design",
        "Operations",
        "Collaboration",
    ]
    answered = 0
    while answered < 3:
        audio, times, values = speech_and_mouth(seed=answered)
        words = [{"word": w, "start": i * 0.4, "end": i * 0.4 + 0.3} for i, w in enumerate(ANSWER.split())]
        body = _answer(
            client,
            sid,
            text=ANSWER,
            words=words,
            audio_wav=wav_b64(audio),
            mouth={"times": times, "values": values},
            gaze_samples=[[i * 0.1, i % 10 != 0] for i in range(60)],
        )
        assert body["turn"] and not body["turn"]["emergency"]
        answered += 1
    st = client.get(f"/sessions/{sid}").json()
    assert all(t["answered"] for t in st["turns"][:-1]) and not st["turns"][-1]["answered"]
    report = client.post(f"/sessions/{sid}/finish").json()
    assert report["summary"]["answers"] == answered and report["summary"]["emergency_questions"] == 0
    assert report["summary"]["label"] == "experimental" and report["company"] == "Acme"
    assert all(a["lipsync"]["decision"] == "match" for a in report["answers"])
    assert all(a["delivery"]["words_per_minute"] > 0 for a in report["answers"])
    assert all(a["method"] == "llm" and a["evidence"] for a in report["answers"])
    assert {s["name"] for s in report["skills"]} == {"System design", "Operations", "Collaboration"}
    assert report["transcript"][0]["speaker"] == "interviewer" and report["moments"]
    assert report["appendix"]["prototype_nine_factor"]["score"] >= 0
    assert client.get("/sessions").json()[0]["overall"] == report["summary"]["overall"]
    assert client.post(f"/sessions/{sid}/turn", json={"text": "late"}).status_code == 409

    link = client.post(f"/reports/{sid}/share", json={"days": 7}).json()["url"]
    token = link.rsplit("/", 1)[1]
    shared = client.get(f"/shared/{token}").json()
    assert "transcript" not in shared and "appendix" not in shared
    client.delete(f"/reports/{sid}/share")
    assert client.get(f"/shared/{token}").status_code == 404


def test_lipsync_mismatch_raises_integrity_notice(client, user, llm):
    sid = _create(client)["id"]
    _answer(client, sid)
    audio, times, values = speech_and_mouth(seed=3, mismatch=True)
    r = _answer(client, sid, text=ANSWER, audio_wav=wav_b64(audio), mouth={"times": times, "values": values})
    assert r["notices"][0]["event"] == "lipsync_mismatch" and not r["notices"][0]["end_session"]


def test_pending_question_is_idempotent(client, user, llm):
    sid = _create(client)["id"]
    a = _answer(client, sid)["turn"]
    b = _answer(client, sid)["turn"]  # no answer given: the same question is returned
    assert a == b and llm.calls == 2  # blueprint + opening only


def test_no_llm_uses_flagged_emergency_questions(client, user):
    sid = _create(client)["id"]
    t = _answer(client, sid)["turn"]
    assert t["emergency"] is True and t["action"] == "open"
    st = client.get(f"/sessions/{sid}").json()
    assert st["blueprint"]["emergency"] is True and st["capabilities"]["llm"] is None


def test_sessions_are_private(client, user, llm):
    sid = _create(client)["id"]
    client.post("/auth/logout")
    signup(client, "other@example.com")
    assert client.get(f"/sessions/{sid}").status_code == 404
    assert client.post(f"/sessions/{sid}/turn", json={}).status_code == 404


def test_phone_debounce_and_proctored_policy(client, user, llm):
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
    assert client.post(f"/sessions/{sid}/turn", json={}).status_code == 409


def test_coaching_mode_never_ends_on_integrity(client, user, llm):
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


def test_coding_challenge_attached_and_graded(client, user, llm):
    sid = _create(client, role="Data Analyst")["id"]
    challenge = client.get(f"/sessions/{sid}").json()["challenge"]
    assert challenge is not None and "tests" not in challenge
    if challenge["languages"] == ["sql"]:
        r = client.post(
            f"/sessions/{sid}/code", json={"language": "sql", "source": "SELECT 1", "final": True}
        ).json()
        assert r["total"] >= 1 and r["passed"] == 0
    else:
        r = client.post(f"/sessions/{sid}/code", json={"language": "python", "source": "print(1)"}).json()
        assert "not configured" in r["error"]
    behavioural = _create(client, role="Data Analyst", interview_type="behavioral")["id"]
    assert client.get(f"/sessions/{behavioural}").json()["challenge"] is None


# ------------------------------------------------------------------ limits and cost caps


def test_plan_session_limit(client, user, monkeypatch):
    monkeypatch.setenv("FREE_SESSIONS_PER_MONTH", "2")
    _create(client)
    _create(client)
    r = client.post("/sessions", data={"role": "Engineer"})
    assert r.status_code == 402


def test_cost_cap_stops_paid_calls_and_flags_emergency(client, user, llm, monkeypatch):
    from interview_api import metering

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
    first = _answer(client, sid)["turn"]
    assert not first["emergency"] and llm.calls == 2
    second = _answer(client, sid, text=ANSWER)["turn"]
    assert second["emergency"] is True and llm.calls == 2  # cap reached: no more paid calls, badge shown


# ------------------------------------------------------------------ privacy


def test_export_and_delete_account(client, user, llm):
    sid = _create(client)["id"]
    _answer(client, sid)
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


@pytest.mark.parametrize("payload", [{"text": "x" * 9000}, {"audio_wav": "!!notbase64!!"}])
def test_input_validation(client, user, llm, payload):
    sid = _create(client)["id"]
    _answer(client, sid)
    assert client.post(f"/sessions/{sid}/turn", json=payload).status_code == 422
