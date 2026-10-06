"""Realtime WebSocket interview with the REAL local speech stack (sherpa-onnx STT + Kokoro TTS).

Only the LLM is scripted here (tests/fake_llm.py); speech recognition and synthesis are the real
models. Skipped when the models are not installed (python -m interview_core.realtime.assets download).
"""

import json
import re

import numpy as np
import pytest
from conftest import signup
from fake_llm import FakeInterviewerLLM
from interview_core.realtime import assets

pytestmark = pytest.mark.skipif(
    not all(assets.is_present(k) for k in ("stt-streaming", "stt-final", "vad", "tts-kokoro")),
    reason="local speech models not installed",
)

SPOKEN = "In my last role I owned the payments service and cut failed transactions by half."


def _speech_pcm16() -> bytes:
    """The candidate's answer: synthesised once with Kokoro (a different voice from the interviewer)."""
    from interview_core.realtime.tts import kokoro_provider

    pcm = b"".join(c.pcm16 for c in kokoro_provider().synthesize(SPOKEN, "michael"))
    x = np.frombuffer(pcm, "<i2").astype(np.float32)
    y = np.interp(np.linspace(0, len(x) - 1, int(len(x) * 16000 / 24000)), np.arange(len(x)), x)
    silence = np.zeros(int(2.5 * 16000))
    return np.concatenate([np.zeros(8000), y, silence]).astype("<i2").tobytes()


@pytest.fixture
def live_client(client, monkeypatch):
    from interview_api import runtime

    fake = FakeInterviewerLLM()
    monkeypatch.setattr(runtime, "llm_chain", lambda: (fake,))
    signup(client)
    sid = client.post("/sessions", data={"role": "Backend Engineer", "duration_minutes": "10"}).json()["id"]
    return client, sid


def _ticket(client, sid):
    r = client.post(f"/sessions/{sid}/realtime")
    assert r.status_code == 200, r.text
    body = r.json()
    assert (
        body["url"].startswith("ws")
        and body["providers"]["stt"] == "sherpa"
        and body["providers"]["tts"] == "kokoro"
    )
    return body["ticket"]


def _until(ws, kind, sink=None, limit=400):
    for _ in range(limit):
        m = ws.receive()
        if m.get("bytes"):
            if sink is not None:
                sink.append(m["bytes"])
            continue
        msg = json.loads(m["text"])
        if sink is not None and msg["type"] != "tts.chunk":
            sink.append(msg)
        if msg["type"] == kind:
            return msg
        if msg["type"] == "error" and not msg.get("recoverable"):
            raise AssertionError(msg)
    raise AssertionError(f"never received {kind}")


def test_spoken_turn_round_trip_with_real_speech(live_client):
    client, sid = live_client
    with client.websocket_connect(f"/ws/interview?ticket={_ticket(client, sid)}") as ws:
        ready = _until(ws, "ready")
        assert ready["providers"]["tts"] == "kokoro" and ready["session"]["persona"]["name"] == "Maya"
        ws.send_text(json.dumps({"type": "start"}))
        q = _until(ws, "question")
        assert q["action"] == "open" and not q["emergency"]
        frames: list = []
        start = _until(ws, "tts.start", frames)
        end = _until(ws, "tts.end", frames)
        audio = b"".join(f[1:] for f in frames if isinstance(f, bytes))
        assert start["sample_rate"] == 24000 and end["duration"] > 1.0
        assert len(audio) / 2 / 24000 == pytest.approx(
            end["duration"], rel=0.05
        )  # interviewer audio really arrived
        assert all(f[0] == q["utterance"] % 256 for f in frames if isinstance(f, bytes))

        ws.send_text(json.dumps({"type": "playback", "state": "ended", "utterance": q["utterance"]}))
        _until(ws, "listening")
        pcm = _speech_pcm16()
        for i in range(0, len(pcm), 3200):  # 100 ms frames
            ws.send_bytes(pcm[i : i + 3200])
        seen: list = []
        _until(ws, "turn.end", seen)
        partials = [m["text"] for m in seen if isinstance(m, dict) and m["type"] == "stt.partial"]
        assert len(partials) >= 3  # live captions while speaking
        final = _until(ws, "stt.final", seen)
        while not final.get("final"):
            final = _until(ws, "stt.final", seen)
        words = lambda s: re.findall(r"[a-z]+", s.lower())  # noqa: E731
        ref, hyp = words(SPOKEN), words(final["text"])
        assert len(set(ref) & set(hyp)) / len(set(ref)) >= 0.9, final["text"]
        diags = []
        nxt = _until(ws, "question", diags)
        assert nxt["index"] == 1 and nxt["utterance"] == q["utterance"] + 1
        stages = {d["stage"] for d in seen + diags if isinstance(d, dict) and d["type"] == "diag"}
        assert {"stt_finalize", "llm"} <= stages

    state = client.get(f"/sessions/{sid}").json()
    assert state["turns"][0]["answered"] and "payments" in state["turns"][0]["answer"].lower()


def test_barge_in_cancels_speech_and_starts_listening(live_client):
    client, sid = live_client
    with client.websocket_connect(f"/ws/interview?ticket={_ticket(client, sid)}") as ws:
        _until(ws, "ready")
        ws.send_text(json.dumps({"type": "start"}))
        q = _until(ws, "question")
        _until(ws, "tts.start")
        ws.send_text(json.dumps({"type": "barge_in"}))
        cancel = _until(ws, "tts.cancel")
        assert cancel["utterance"] == q["utterance"]
        _until(ws, "listening")


def test_server_detects_barge_in_and_keeps_the_interrupting_words(live_client):
    """No client barge_in message: the server's VAD hears the candidate over the interviewer,
    cancels the question audio and starts the turn with the words that interrupted it."""
    client, sid = live_client
    with client.websocket_connect(f"/ws/interview?ticket={_ticket(client, sid)}") as ws:
        _until(ws, "ready")
        ws.send_text(json.dumps({"type": "start"}))
        q = _until(ws, "question")
        _until(ws, "tts.start")
        pcm = _speech_pcm16()
        seen: list = []
        sent = 0
        for i in range(0, len(pcm), 3200):  # stream the answer in 100 ms frames
            ws.send_bytes(pcm[i : i + 3200])
            sent = i + 3200
            if i == 3200 * 12:  # after 1.2 s, the server should have noticed
                break
        barge = _until(ws, "barge_in", seen)
        assert barge["source"] == "server"
        assert any(
            isinstance(m, dict) and m["type"] == "tts.cancel" and m["utterance"] == q["utterance"]
            for m in seen
        )
        _until(ws, "listening", seen)
        for i in range(sent, len(pcm), 3200):
            ws.send_bytes(pcm[i : i + 3200])
        final = _until(ws, "stt.final", seen)
        while not final.get("final"):
            final = _until(ws, "stt.final", seen)
        hyp = re.findall(r"[a-z]+", final["text"].lower())
        assert hyp[:3] == ["in", "my", "last"], final["text"]  # the first words were not lost


def test_socket_stays_responsive_while_a_slow_llm_thinks(client, monkeypatch):
    """A slow LLM (CPU-only local models take 30-60 s) must not stall the socket: mic frames and
    pings are still read and answered while the turn is being generated."""
    import time as _time

    from interview_api import runtime

    class SlowLLM(FakeInterviewerLLM):
        def stream(self, *a, **kw):
            _time.sleep(3)
            yield from super().stream(*a, **kw)

    slow = SlowLLM()
    monkeypatch.setattr(runtime, "llm_chain", lambda: (slow,))
    signup(client)
    sid = client.post("/sessions", data={"role": "Backend Engineer", "duration_minutes": "10"}).json()["id"]
    with client.websocket_connect(f"/ws/interview?ticket={_ticket(client, sid)}") as ws:
        _until(ws, "ready")
        ws.send_text(json.dumps({"type": "start"}))
        _until(ws, "thinking")
        for _ in range(60):  # 6 s of (silent) mic frames while the LLM is busy
            ws.send_bytes(b"\x00\x00" * 1600)
        ws.send_text(json.dumps({"type": "ping", "t": 1}))
        seen: list = []
        _until(ws, "pong", seen)
        assert not any(
            isinstance(m, dict) and m["type"] == "question" for m in seen
        )  # answered while thinking
        _until(ws, "question")


def test_tickets_are_single_use_and_signed(live_client):
    client, sid = live_client
    t = _ticket(client, sid)
    with client.websocket_connect(f"/ws/interview?ticket={t}") as ws:
        _until(ws, "ready")
    from starlette.websockets import WebSocketDisconnect

    with pytest.raises(WebSocketDisconnect) as e, client.websocket_connect(f"/ws/interview?ticket={t}") as ws:
        ws.receive_text()
    assert e.value.code == 4401
    with (
        pytest.raises(WebSocketDisconnect) as e2,
        client.websocket_connect("/ws/interview?ticket=forged") as ws,
    ):
        ws.receive_text()
    assert e2.value.code == 4401


def test_reconnect_resumes_the_pending_question(live_client):
    client, sid = live_client
    with client.websocket_connect(f"/ws/interview?ticket={_ticket(client, sid)}") as ws:
        _until(ws, "ready")
        ws.send_text(json.dumps({"type": "start"}))
        first = _until(ws, "question")
        _until(ws, "tts.end")
    with client.websocket_connect(f"/ws/interview?ticket={_ticket(client, sid)}") as ws:
        ready = _until(ws, "ready")
        assert ready["session"]["current"]["say"] == first["say"]
        ws.send_text(json.dumps({"type": "start"}))
        again = _until(ws, "question")
        assert again["say"] == first["say"] and again["index"] == 0  # same question, no new LLM turn
