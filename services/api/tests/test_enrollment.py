import numpy as np
import pytest
from conftest import signup
from helpers import liveness_series, png_b64, tone, wav_b64


class ColourEmbedder:
    model_id = "test-colour"

    def embed(self, face):
        return face.reshape(-1, 3).mean(0) + 1.0


class PitchEmbedder:
    model_id = "test-pitch"

    def embed(self, audio, sr):
        spec = np.abs(np.fft.rfft(audio[:sr]))[:1000]
        return spec / (np.linalg.norm(spec) + 1e-9)


@pytest.fixture
def biometric(client, monkeypatch):
    from interview_api import runtime

    monkeypatch.setattr(runtime, "face_embedder", lambda: ColourEmbedder())
    monkeypatch.setattr(runtime, "speaker_embedder", lambda: PitchEmbedder())
    monkeypatch.setenv("FACE_MATCH_THRESHOLD", "0.99")
    monkeypatch.setenv("VOICE_MATCH_THRESHOLD", "0.8")
    return signup(client, consents=("data_processing", "biometric_face", "biometric_voice"))


def test_face_enrolment_requires_consent_and_configuration(client):
    signup(client)
    assert client.post("/enrollment/liveness/challenge").status_code == 403
    client.post("/consent", json={"kind": "biometric_face", "granted": True})
    ch = client.post("/enrollment/liveness/challenge").json()
    r = client.post(
        "/enrollment/face",
        json={
            "nonce": ch["nonce"],
            "series": liveness_series(ch["nonce"], ch["steps"]),
            "faces": [png_b64(seed=i) for i in range(5)],
        },
    )
    assert r.status_code == 503  # no licensed face model configured: honest refusal, no fake result


def test_face_enrolment_liveness_and_replay(client, biometric):
    ch = client.post("/enrollment/liveness/challenge").json()
    series = liveness_series(ch["nonce"], ch["steps"])
    faces = [png_b64(seed=i) for i in range(6)]
    r = client.post("/enrollment/face", json={"nonce": ch["nonce"], "series": series, "faces": faces})
    assert r.status_code == 200, r.text
    assert client.get("/enrollment/status").json()["face"]
    # Same nonce again -> used; same series under a new nonce -> replay digest.
    assert (
        client.post(
            "/enrollment/face", json={"nonce": ch["nonce"], "series": series, "faces": faces}
        ).status_code
        == 422
    )
    ch2 = client.post("/enrollment/liveness/challenge").json()
    replay = {**series, "nonce": ch2["nonce"]}
    assert (
        client.post(
            "/enrollment/face", json={"nonce": ch2["nonce"], "series": replay, "faces": faces}
        ).status_code
        == 422
    )
    # Static photo fails liveness.
    ch3 = client.post("/enrollment/liveness/challenge").json()
    photo = liveness_series(ch3["nonce"], [])
    assert (
        client.post(
            "/enrollment/face", json={"nonce": ch3["nonce"], "series": photo, "faces": faces}
        ).status_code
        == 422
    )


def test_template_is_encrypted_at_rest(client, biometric):
    from interview_api.db import get_db
    from interview_api.models import BiometricTemplate

    ch = client.post("/enrollment/liveness/challenge").json()
    client.post(
        "/enrollment/face",
        json={
            "nonce": ch["nonce"],
            "series": liveness_series(ch["nonce"], ch["steps"]),
            "faces": [png_b64(seed=i) for i in range(6)],
        },
    )
    row = next(get_db()).query(BiometricTemplate).one()
    assert "ciphertext" in row.blob and "201" not in row.blob


def test_session_face_check_match_and_mismatch(client, biometric):
    ch = client.post("/enrollment/liveness/challenge").json()
    client.post(
        "/enrollment/face",
        json={
            "nonce": ch["nonce"],
            "series": liveness_series(ch["nonce"], ch["steps"]),
            "faces": [png_b64(seed=i) for i in range(6)],
        },
    )
    sid = client.post("/sessions", data={"role": "Engineer"}).json()["id"]
    assert (
        client.post(f"/sessions/{sid}/face-check", json={"image": png_b64(seed=9)}).json()["status"]
        == "match"
    )
    other = png_b64(color=(40, 40, 220))
    res = [client.post(f"/sessions/{sid}/face-check", json={"image": other, "t": i}).json() for i in range(3)]
    assert res[0]["status"] == "mismatch" and res[2]["notice"]["event"] == "face_mismatch"


def test_voice_enrolment_with_phrases_then_matching(client, biometric):
    recs = []
    for i in range(3):
        p = client.post("/enrollment/voice/phrase").json()
        recs.append(
            {"nonce": p["nonce"], "audio_wav": wav_b64(tone(220, seed=i)), "client_transcript": p["phrase"]}
        )
    r = client.post("/enrollment/voice", json={"recordings": recs})
    assert r.status_code == 200, r.text
    assert all(c["checked_by"] == "client_reported" for c in r.json()["phrases"])
    assert client.post("/enrollment/voice", json={"recordings": recs}).status_code == 422  # nonces used
    sid = client.post("/sessions", data={"role": "Engineer"}).json()["id"]
    client.post(f"/sessions/{sid}/next")
    same = client.post(
        f"/sessions/{sid}/answer", json={"transcript": "hello", "audio_wav": wav_b64(tone(220, seed=7))}
    ).json()
    assert same["signals"]["voice"]["status"] == "match"
    client.post(f"/sessions/{sid}/next")
    diff = client.post(
        f"/sessions/{sid}/answer", json={"transcript": "hello", "audio_wav": wav_b64(tone(500, seed=7))}
    ).json()
    assert (
        diff["signals"]["voice"]["status"] == "mismatch" and diff["notices"][0]["event"] == "voice_mismatch"
    )


def test_wrong_phrase_rejected_and_consent_withdrawal_deletes_template(client, biometric):
    p = client.post("/enrollment/voice/phrase").json()
    bad = [
        {"nonce": p["nonce"], "audio_wav": wav_b64(tone(220)), "client_transcript": "something else entirely"}
    ] * 3
    assert client.post("/enrollment/voice", json={"recordings": bad}).status_code == 422
    recs = []
    for i in range(3):
        p = client.post("/enrollment/voice/phrase").json()
        recs.append(
            {"nonce": p["nonce"], "audio_wav": wav_b64(tone(220, seed=i)), "client_transcript": p["phrase"]}
        )
    client.post("/enrollment/voice", json={"recordings": recs})
    assert client.get("/enrollment/status").json()["voice"]
    client.post("/consent", json={"kind": "biometric_voice", "granted": False})
    assert not client.get("/enrollment/status").json()["voice"]
