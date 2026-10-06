import base64
import os

import pytest

os.environ.setdefault("TEMPLATE_KEK_BASE64", base64.b64encode(b"k" * 32).decode())
os.environ["DATABASE_URL"] = "sqlite:///:memory:"
os.environ["SPEECH_WARMUP"] = "0"  # models load on first use (only the realtime tests need them)
for var in (
    "LLM_PROVIDER",
    "ASR_PROVIDER",
    "TTS_PROVIDER",
    "FACE_EMBEDDER",
    "SPEAKER_EMBEDDER",
    "JUDGE0_URL",
):
    os.environ.pop(var, None)


@pytest.fixture
def app(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path}/test.db")
    monkeypatch.setenv("ADMIN_EMAILS", "admin@example.com")
    from interview_api import metering, ratelimit, runtime
    from interview_api.db import reset_engine
    from interview_api.settings import get_settings

    get_settings.cache_clear()
    metering.prices.cache_clear()
    runtime.clear()
    ratelimit.reset_all()
    reset_engine()
    from interview_api.main import create_app

    yield create_app()
    reset_engine()
    get_settings.cache_clear()


@pytest.fixture
def client(app):
    from fastapi.testclient import TestClient

    with TestClient(app, headers={"x-ic-csrf": "1"}) as c:
        yield c


def signup(client, email="user@example.com", consents=("data_processing",)):
    r = client.post(
        "/auth/register",
        json={
            "email": email,
            "password": "correct horse battery",
            "name": "Test",
            "accept_terms": True,
            "age_confirmed": True,
        },
    )
    assert r.status_code == 201, r.text
    for k in consents:
        assert client.post("/consent", json={"kind": k, "granted": True}).status_code == 200
    return r.json()


@pytest.fixture
def user(client):
    return signup(client)
