import base64
import os
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest
from cryptography.exceptions import InvalidTag
from interview_core.biometric import build_template, verify
from interview_core.crypto import LocalKeyProvider, decrypt_template, encrypt_template, is_expired
from interview_core.face import FaceVerifier, assess_quality
from interview_core.voice import VoiceSession, VoiceStatus, VoiceVerifier, check_phrase, issue_phrase


class PitchEmbedder:
    """Test double: embeds audio by its normalised magnitude spectrum below 1 kHz."""

    model_id = "test-pitch"

    def embed(self, audio, sr):
        spec = np.abs(np.fft.rfft(audio[:sr]))[:1000]
        return spec / (np.linalg.norm(spec) + 1e-9)


class MeanColourEmbedder:
    model_id = "test-colour"

    def embed(self, face):
        return face.reshape(-1, 3).mean(0) + 1.0


def tone(freq, seconds=3.0, sr=16000, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * sr)) / sr
    return (0.3 * np.sin(2 * np.pi * freq * t) + 0.01 * rng.normal(0, 1, t.size)).astype(np.float32)


def test_template_drops_outliers_and_requires_enough_samples():
    good = [np.array([1.0, 0.1 * i, 0]) for i in range(5)]
    tpl = build_template("face", "m", [*good, np.array([-1.0, 0, 0])], min_samples=5)
    assert tpl.n_samples == 5
    with pytest.raises(ValueError):
        build_template("face", "m", good[:4], min_samples=5)
    with pytest.raises(ValueError, match="re-enrol"):
        verify(tpl, good[0], 0.5, "other-model")
    assert verify(tpl, good[0], None, "m").accepted is None


def test_voice_session_never_matches_without_enrolment():
    session = VoiceSession(VoiceVerifier(PitchEmbedder(), 0.8), template=None)
    assert session.check(tone(200), 16000).status == VoiceStatus.NOT_ENROLLED
    assert session.template is None  # the prototype would have adopted this answer


def test_voice_enrol_match_mismatch_short_uncalibrated():
    v = VoiceVerifier(PitchEmbedder(), 0.8)
    tpl = v.enrol([(tone(200, seed=i), 16000) for i in range(3)])
    s = VoiceSession(v, tpl)
    assert s.check(tone(200, seed=9), 16000).status == VoiceStatus.MATCH
    assert s.check(tone(450, seed=9), 16000).status == VoiceStatus.MISMATCH
    assert s.check(tone(200, seconds=1.0), 16000).status == VoiceStatus.TOO_SHORT
    assert s.mismatch_count == 1
    unc = VoiceSession(VoiceVerifier(PitchEmbedder(), None), tpl)
    assert unc.check(tone(200), 16000).status == VoiceStatus.UNCALIBRATED
    with pytest.raises(ValueError, match="enrolment needs"):
        v.enrol([(tone(200, seconds=1.0), 16000)] * 3)


def test_phrase_challenge():
    a, b = issue_phrase(), issue_phrase()
    assert a.phrase != b.phrase and len(a.phrase.split()) == 6
    words = a.phrase.split()
    assert check_phrase(a, a.phrase.upper() + ".")[0]
    assert check_phrase(a, " ".join([*words[:-1], "tree"]))[0]  # one ASR error tolerated
    assert not check_phrase(a, "hello this is a recording")[0]
    assert not check_phrase(a, a.phrase, now=a.expires_at + timedelta(seconds=1))[0]


def test_template_encryption_roundtrip_and_binding():
    keys = LocalKeyProvider(os.urandom(32))
    tpl = build_template("voice", "m1", [np.array([1.0, 2.0, 3.0])] * 3, min_samples=3)
    blob = encrypt_template(tpl, "user-1", keys)
    assert "1.0" not in blob.ciphertext
    back = decrypt_template(type(blob).from_json(blob.to_json()), "user-1", "voice", "m1", keys)
    assert np.allclose(back.vector, tpl.vector) and back.n_samples == 3
    with pytest.raises(InvalidTag):
        decrypt_template(blob, "user-2", "voice", "m1", keys)  # copied to another account
    with pytest.raises(InvalidTag):
        decrypt_template(blob, "user-1", "voice", "m1", LocalKeyProvider(os.urandom(32)))


def test_kek_must_be_configured(monkeypatch):
    monkeypatch.delenv("TEMPLATE_KEK_BASE64", raising=False)
    with pytest.raises(RuntimeError, match="refusing"):
        LocalKeyProvider.from_env()
    monkeypatch.setenv("TEMPLATE_KEK_BASE64", base64.b64encode(os.urandom(32)).decode())
    assert LocalKeyProvider.from_env().key_id == "local-v1"


def test_retention():
    now = datetime(2026, 1, 31, tzinfo=UTC)
    assert is_expired(datetime(2026, 1, 1, tzinfo=UTC), 30, now)
    assert not is_expired(datetime(2026, 1, 15, tzinfo=UTC), 30, now)
    assert is_expired(now, 0, now)


def test_face_verifier_and_quality():
    rng = np.random.default_rng(0)
    person = [
        np.full((112, 112, 3), (200, 50, 50), np.uint8) + rng.integers(0, 5, (112, 112, 3)).astype(np.uint8)
        for _ in range(6)
    ]
    fv = FaceVerifier(MeanColourEmbedder(), threshold=0.99)
    tpl = fv.enrol(person)
    assert fv.verify(tpl, person[0]).accepted
    assert not fv.verify(tpl, np.full((112, 112, 3), (50, 50, 200), np.uint8)).accepted
    sharp = (rng.integers(0, 2, (112, 112, 3)) * 255).astype(np.uint8)
    assert assess_quality(sharp, 200, 640, 1).ok
    flat = np.full((112, 112, 3), 128, np.uint8)
    rep = assess_quality(flat, 50, 640, 2)
    assert not rep.ok and len(rep.problems) == 3


def test_gcp_kms_provider_wraps_through_kms_client():
    from types import SimpleNamespace

    from interview_core.crypto import GcpKmsKeyProvider

    class FakeKms:
        def __init__(self):
            self.calls = []

        def encrypt(self, request):
            self.calls.append(("encrypt", request["name"]))
            return SimpleNamespace(ciphertext=b"W" + request["plaintext"])

        def decrypt(self, request):
            self.calls.append(("decrypt", request["name"]))
            return SimpleNamespace(plaintext=request["ciphertext"][1:])

    kms = FakeKms()
    keys = GcpKmsKeyProvider("projects/p/locations/asia-south1/keyRings/r/cryptoKeys/templates", client=kms)
    tpl = build_template("face", "m", [np.array([1.0, 0.0])] * 5, min_samples=5)
    blob = encrypt_template(tpl, "u", keys)
    assert blob.kek_id.endswith("cryptoKeys/templates")
    assert np.allclose(decrypt_template(blob, "u", "face", "m", keys).vector, tpl.vector)
    assert [c[0] for c in kms.calls] == ["encrypt", "decrypt"]
