"""Realtime spoken interview over one WebSocket per session.

Protocol (JSON text frames plus binary audio frames):

Client -> server
  binary                    16 kHz mono PCM16LE microphone frames (sent continuously)
  {"type": "start"}         begin, or resume after a reconnect (re-asks the pending question)
  {"type": "playback", "state": "ended", "utterance": n}   the interviewer audio finished playing
  {"type": "barge_in"}      the candidate started talking over the interviewer
  {"type": "answer_meta", "mouth": {...}, "gaze": [...]}   E4/E5 series for the turn just ended
  {"type": "text_answer", "text": "..."}   typed answer instead of speech
  {"type": "repeat"} | {"type": "skip"} | {"type": "end"} | {"type": "ping", "t": ...}

Server -> client
  ready, phase, stt.partial, stt.final, turn.end, thinking, question, tts.start, tts.chunk,
  tts.end, tts.cancel, barge_in (server-detected), listening, notice, diag, error, done, pong
  binary: 1 byte (utterance mod 256) + PCM16LE interviewer audio at tts.start's sample_rate

Turn-taking: the candidate's turn ends after a pause (shorter when the transcript so far ends a
sentence, longer when it trails off on "and", "so", "um"...), a hard cap of 4 minutes, a typed
answer or a skip. Audio is only fed to STT while listening; barge-in is detected client-side.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import hashlib
import hmac
import json
import logging
import os
import queue
import re
import secrets
import threading
import time
from typing import Any

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Request, WebSocket
from interview_core.agent.personas import persona
from interview_core.agent.state import AgentState
from interview_core.realtime.stt import STTEvent, TurnTranscript
from interview_core.speech.asr import Word
from sqlalchemy.orm import Session
from starlette.websockets import WebSocketDisconnect, WebSocketState

from interview_api import interview, runtime
from interview_api.db import get_db, session_scope
from interview_api.models import InterviewSession, User
from interview_api.security import current_user

log = logging.getLogger("interview_api.live")
router = APIRouter(tags=["realtime"])

TICKET_TTL_S = 60
MAX_ANSWER_S = 240.0
SILENCE_END_MS = float(os.getenv("TURN_SILENCE_MS", "1000"))
# Detect barge-in on the server too (Silero VAD on the mic stream while the interviewer speaks).
SERVER_BARGE_IN = os.getenv("SERVER_BARGE_IN", "1") != "0"
SLOW_COMMANDS = {"start", "text_answer", "skip", "repeat"}  # "end" stays immediate
SILENCE_TRAILING_MS = float(os.getenv("TURN_SILENCE_TRAILING_MS", "1800"))
_TRAILING = re.compile(r"\b(and|but|so|because|or|um+|uh+|like|then|which|that|the|a|to|of|with)\W*$", re.I)
_SECRET = os.getenv("REALTIME_SECRET", "").encode() or secrets.token_bytes(32)
_used_nonces: dict[str, float] = {}
_live: dict[str, Conversation] = {}
# Interviewer turns in flight per session. A turn outlives a dropped connection; a reconnect waits
# for it instead of generating a second one.
_advancing: dict[str, asyncio.Future] = {}


# ----------------------------------------------------------------------------- tickets


def make_ticket(session_id: str, user_id: str, now: float | None = None) -> str:
    exp = int((now or time.time()) + TICKET_TTL_S)
    payload = f"{session_id}.{user_id}.{exp}.{secrets.token_hex(8)}"
    sig = hmac.new(_SECRET, payload.encode(), hashlib.sha256).hexdigest()
    return base64.urlsafe_b64encode(f"{payload}.{sig}".encode()).decode().rstrip("=")


def check_ticket(ticket: str, now: float | None = None) -> tuple[str, str]:
    now = now or time.time()
    try:
        raw = base64.urlsafe_b64decode(ticket + "=" * (-len(ticket) % 4)).decode()
        session_id, user_id, exp, nonce, sig = raw.split(".")
    except Exception as e:
        raise PermissionError("malformed ticket") from e
    expected = hmac.new(_SECRET, f"{session_id}.{user_id}.{exp}.{nonce}".encode(), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(sig, expected):
        raise PermissionError("bad ticket signature")
    if int(exp) < now:
        raise PermissionError("ticket expired")
    for k, v in list(_used_nonces.items()):
        if v < now:
            del _used_nonces[k]
    if nonce in _used_nonces:
        raise PermissionError("ticket already used")
    _used_nonces[nonce] = int(exp)
    return session_id, user_id


def _ws_base(request: Request) -> str:
    public = os.getenv("PUBLIC_API_URL", "").rstrip("/")
    base = public or str(request.base_url).rstrip("/")
    return re.sub(r"^http", "ws", base)


@router.post("/sessions/{session_id}/realtime")
def realtime_ticket(
    session_id: str, request: Request, user: User = Depends(current_user), db: Session = Depends(get_db)
) -> dict:
    s = db.get(InterviewSession, session_id)
    if s is None or s.user_id != user.id:
        raise HTTPException(404, "session not found")
    if s.status != "active":
        raise HTTPException(409, f"session is {s.status.replace('_', ' ')}")
    caps = runtime.capabilities()
    return {
        "ticket": make_ticket(session_id, user.id),
        "url": f"{_ws_base(request)}/ws/interview",
        "expires_in": TICKET_TTL_S,
        "providers": {k: caps[k] for k in ("llm", "llm_model", "llm_fallback", "stt", "tts")},
    }


# ----------------------------------------------------------------------------- conversation

_STOP = object()


class Conversation:
    def __init__(self, ws: WebSocket, session_id: str, user_id: str):
        self.ws, self.session_id, self.user_id = ws, session_id, user_id
        self.loop = asyncio.get_running_loop()
        self.phase = "idle"
        self.utterance = 0
        self.turn_id = 0  # increments per candidate turn; stale STT results are dropped
        self.tts_task: asyncio.Task | None = None
        self.playback_timer: asyncio.Task | None = None
        self.audio_q: queue.Queue = queue.Queue()
        self.turn_audio = bytearray()
        self.listen_started = 0.0
        self.last_partial = ""
        self.end_of_speech_at = 0.0
        self.meta_event = asyncio.Event()
        self.meta: dict[str, Any] = {}
        self.closing_turn = False
        self.audio_out = False  # interviewer audio is being streamed for the current utterance
        self.recent_in = bytearray()  # last 2.5 s of mic audio while the interviewer speaks
        self.cmd_q: asyncio.Queue = asyncio.Queue()
        self.consumer: asyncio.Task | None = None
        self.send_lock = asyncio.Lock()
        self.stt_provider = runtime.stt_provider()
        self.tts_provider = runtime.tts_provider()
        self.worker = threading.Thread(target=self._stt_worker, name=f"stt-{session_id[:8]}", daemon=True)
        self.language = "en"
        self.voice = None

    # ---- io
    async def send(self, obj: dict) -> None:
        if self.ws.application_state != WebSocketState.CONNECTED:
            return
        async with self.send_lock:
            with contextlib.suppress(Exception):
                await self.ws.send_text(json.dumps(obj))

    async def send_audio(self, utt: int, pcm: bytes) -> None:
        if self.ws.application_state != WebSocketState.CONNECTED:
            return
        async with self.send_lock:
            step = 32000  # ~0.66 s at 24 kHz per frame keeps frames small
            for i in range(0, len(pcm), step):
                with contextlib.suppress(Exception):
                    await self.ws.send_bytes(bytes([utt % 256]) + pcm[i : i + step])

    async def set_phase(self, phase: str) -> None:
        self.phase = phase
        await self.send({"type": "phase", "phase": phase})

    async def diag(self, stage: str, ms: float, **extra: Any) -> None:
        await self.send({"type": "diag", "stage": stage, "ms": round(ms, 1), **extra})

    # ---- db helpers (run in threads)
    def _load(self) -> tuple[AgentState, InterviewSession]:
        with session_scope() as db:
            s = db.get(InterviewSession, self.session_id)
            return AgentState.from_dict(dict(s.state)), s

    def _snapshot(self) -> dict:
        with session_scope() as db:
            s = db.get(InterviewSession, self.session_id)
            st = AgentState.from_dict(dict(s.state))
            return {
                "status": s.status,
                "finished": st.finished,
                "turns": [interview.public_turn(st, t) for t in st.turns],
                "current": interview.public_turn(st, st.awaiting_answer),
                "persona": persona(st.params.persona).public(),
                "language": st.params.language,
                "remaining_s": st.remaining_s() if st.started_at else st.params.duration_minutes * 60,
            }

    def _advance(
        self, text: str | None, words: list[Word], seconds: float, source: str, audio: bytes, meta: dict
    ):
        with session_scope() as db:
            s = db.get(InterviewSession, self.session_id)
            user = db.get(User, self.user_id)
            pcm = np.frombuffer(bytes(audio), dtype="<i2").astype(np.float32) / 32768.0 if audio else None
            turn, st, notices = interview.advance(
                db,
                user,
                s,
                text,
                words=words,
                seconds=seconds,
                source=source,
                audio=pcm if pcm is not None and pcm.size > 1600 else None,
                sr=16000,
                mouth=meta.get("mouth"),
                gaze=meta.get("gaze"),
            )
            return (interview.public_turn(st, turn) if turn else None), st.finished, notices, s.status

    # ---- STT worker thread
    def _post(self, coro) -> None:
        asyncio.run_coroutine_threadsafe(coro, self.loop)

    def _stt_worker(self) -> None:
        sess = None
        turn = -1
        gate, gate_utt = None, -1
        while True:
            item = self.audio_q.get()
            if item is _STOP:
                break
            try:
                if isinstance(item, tuple) and item[0] == "begin":
                    if sess is not None:
                        sess.close()
                    sess, turn = self.stt_provider.open(self.language), item[1]
                elif isinstance(item, tuple) and item[0] == "finalize":
                    if sess is not None and item[1] == turn:
                        t0 = time.monotonic()
                        events, tr = sess.finalize()
                        sess.close()
                        sess = None
                        for e in events:
                            self._post(self.on_stt_event(e, turn))
                        self._post(self.on_transcript(tr, turn, (time.monotonic() - t0) * 1000))
                elif isinstance(item, tuple) and item[0] == "gate":
                    gate, gate_utt = (self.stt_provider.gate() if self.stt_provider else None), item[1]
                elif isinstance(item, tuple) and item[0] == "gate_audio":
                    if gate is not None and gate.accept(item[1]):
                        self._post(self.server_barge_in(gate_utt))
                        gate = None
                elif isinstance(item, tuple) and item[0] == "discard":
                    if sess is not None:
                        sess.close()
                    sess = None
                elif isinstance(item, bytes | bytearray) and sess is not None:
                    for e in sess.accept(bytes(item)):
                        self._post(self.on_stt_event(e, turn))
                    self._post(self.on_vad(sess.heard_speech, sess.silence_ms, turn))
            except Exception as e:  # provider failure: report, keep the session alive
                log.exception("stt failure")
                sess = None
                self._post(
                    self.send(
                        {"type": "error", "code": "stt_failed", "message": str(e)[:200], "recoverable": True}
                    )
                )

    # ---- STT callbacks (event loop)
    async def on_stt_event(self, e: STTEvent, turn: int) -> None:
        if turn != self.turn_id:
            return
        if e.kind == "partial":
            self.last_partial = e.text
            await self.send({"type": "stt.partial", "text": e.text})
        else:
            await self.send({"type": "stt.final", "text": e.text, "segment": e.segment})

    async def on_vad(self, heard: bool, silence_ms: float, turn: int) -> None:
        if turn != self.turn_id or self.phase != "listening":
            return
        elapsed = time.monotonic() - self.listen_started
        limit = SILENCE_TRAILING_MS if _TRAILING.search(self.last_partial or "") else SILENCE_END_MS
        if (heard and silence_ms >= limit) or elapsed >= MAX_ANSWER_S:
            self.end_of_speech_at = time.monotonic() - silence_ms / 1000.0
            await self.end_turn()

    async def on_transcript(self, tr: TurnTranscript, turn: int, finalize_ms: float) -> None:
        if turn != self.turn_id:
            return
        await self.diag("stt_finalize", finalize_ms, provider=tr.provider)
        await self.send({"type": "stt.final", "text": tr.text, "segment": -1, "final": True})
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(self.meta_event.wait(), timeout=0.6)
        await self.respond(tr.text, tr.words, tr.audio_seconds, tr.provider)

    # ---- turn control
    async def begin_listening(self, preroll: bytes = b"") -> None:
        self.audio_out = False
        self.recent_in = bytearray()
        if self.stt_provider is None:
            await self.set_phase("listening")
            return
        self.turn_id += 1
        self.turn_audio = bytearray()
        self.last_partial = ""
        self.meta_event.clear()
        self.meta = {}
        self.audio_q.put(("begin", self.turn_id))
        if preroll:  # speech that triggered a server-side barge-in: the turn starts with it
            self.turn_audio.extend(preroll)
            self.audio_q.put(preroll)
        self.listen_started = time.monotonic()
        await self.set_phase("listening")
        await self.send({"type": "listening", "turn": self.turn_id})

    async def end_turn(self) -> None:
        if self.phase != "listening":
            return
        await self.set_phase("thinking")
        await self.send({"type": "turn.end", "turn": self.turn_id})
        self.audio_q.put(("finalize", self.turn_id))

    async def respond(self, text: str | None, words: list[Word], seconds: float, source: str) -> None:
        """Record the answer (None = start) and speak the next question."""
        await self.set_phase("thinking")
        await self.send({"type": "thinking"})
        t0 = time.monotonic()
        audio, meta = bytes(self.turn_audio), dict(self.meta)
        inflight = _advancing.get(self.session_id)
        if inflight is not None and not inflight.done() and text is None:
            # (re)start while a previous connection's turn is still being generated: wait for it
            with contextlib.suppress(Exception):
                await asyncio.shield(inflight)
            snap = await asyncio.to_thread(self._snapshot)
            if snap["current"]:
                await self.speak(snap["current"])
            elif snap["finished"] or snap["status"] != "active":
                await self.finish()
            return
        try:
            fut = asyncio.ensure_future(
                asyncio.to_thread(self._advance, text, words, seconds, source, audio, meta)
            )
            _advancing[self.session_id] = fut
            turn, finished, notices, status = await asyncio.shield(fut)
        except Exception as e:
            log.exception("interviewer turn failed")
            await self.send(
                {"type": "error", "code": "turn_failed", "message": str(e)[:200], "recoverable": True}
            )
            await self.begin_listening()
            return
        done_fut = _advancing.get(self.session_id)
        if done_fut is not None and done_fut.done():
            _advancing.pop(self.session_id, None)
        await self.diag("llm", (time.monotonic() - t0) * 1000)
        if self.phase == "done":  # ended while the turn was being generated
            return
        for n in notices:
            await self.send({"type": "notice", **n})
        if turn is None or status != "active":
            await self.finish()
            return
        self.closing_turn = finished and turn["action"] == "close"
        await self.speak(turn)

    async def speak(self, turn: dict) -> None:
        self.utterance += 1
        utt = self.utterance
        await self.set_phase("speaking")
        await self.send({"type": "question", "utterance": utt, **turn})
        self.tts_task = asyncio.create_task(self._tts(turn["say"], utt))

    async def _tts(self, text: str, utt: int) -> None:
        if self.tts_provider is None:
            await self.send({"type": "tts.end", "utterance": utt, "audio": False})
            self._arm_playback_timer(utt, 1.0 + len(text) / 15)
            return
        t0 = time.monotonic()
        gen = self.tts_provider.synthesize(text, self.voice)
        total_s, first, offset = 0.0, True, 0.0
        try:
            while True:
                chunk = await asyncio.to_thread(next, gen, None)
                if chunk is None or utt != self.utterance:
                    break
                if first:
                    first = False
                    if SERVER_BARGE_IN and not self.closing_turn:
                        self.audio_out = True
                        self.audio_q.put(("gate", utt))
                    await self.send(
                        {
                            "type": "tts.start",
                            "utterance": utt,
                            "sample_rate": chunk.sample_rate,
                            "provider": self.tts_provider.name,
                        }
                    )
                    await self.diag(
                        "tts_first_audio", (time.monotonic() - t0) * 1000, provider=self.tts_provider.name
                    )
                    if self.end_of_speech_at:
                        await self.diag("turn_gap", (time.monotonic() - self.end_of_speech_at) * 1000)
                        self.end_of_speech_at = 0.0
                dur = len(chunk.pcm16) / 2 / chunk.sample_rate
                await self.send(
                    {
                        "type": "tts.chunk",
                        "utterance": utt,
                        "text": chunk.text,
                        "offset": round(offset, 3),
                        "duration": round(dur, 3),
                        "marks": [
                            {"t": round(offset + m.t, 3), "kind": m.kind, "value": m.value}
                            for m in chunk.marks
                        ],
                    }
                )
                await self.send_audio(utt, chunk.pcm16)
                offset += dur
                total_s += dur
        except Exception as e:
            log.exception("tts failure")
            await self.send(
                {"type": "error", "code": "tts_failed", "message": str(e)[:200], "recoverable": True}
            )
        if utt == self.utterance:
            await self.send(
                {"type": "tts.end", "utterance": utt, "audio": total_s > 0, "duration": round(total_s, 3)}
            )
            self._arm_playback_timer(utt, total_s + 3.0)

    def _arm_playback_timer(self, utt: int, seconds: float) -> None:
        """If the client never reports playback end (audio blocked, tab hidden), move on anyway."""
        if self.playback_timer:
            self.playback_timer.cancel()

        async def fire():
            await asyncio.sleep(seconds)
            await self.on_playback_ended(utt)

        self.playback_timer = asyncio.create_task(fire())

    async def on_playback_ended(self, utt: int) -> None:
        if utt != self.utterance or self.phase != "speaking":
            return
        if self.playback_timer:
            self.playback_timer.cancel()
            self.playback_timer = None
        if self.closing_turn:
            await self.finish()
        else:
            await self.begin_listening()

    async def barge_in(self) -> None:
        if self.phase != "speaking" or self.closing_turn:
            return
        self.utterance += 1  # invalidates the running TTS stream
        if self.tts_task:
            self.tts_task.cancel()
        await self.send({"type": "tts.cancel", "utterance": self.utterance - 1})
        await self.begin_listening(bytes(self.recent_in))  # keep the words that interrupted

    async def server_barge_in(self, utt: int) -> None:
        """The server's VAD heard the candidate talk over the interviewer (works even when the
        browser's own VAD lags, e.g. on a busy main thread)."""
        if utt != self.utterance or self.phase != "speaking" or self.closing_turn:
            return
        self.utterance += 1
        if self.tts_task:
            self.tts_task.cancel()
        await self.send({"type": "tts.cancel", "utterance": self.utterance - 1})
        await self.send({"type": "barge_in", "source": "server"})
        # recent_in holds every frame up to this moment, including any that arrived after the VAD fired
        await self.begin_listening(bytes(self.recent_in))

    async def finish(self) -> None:
        await self.set_phase("done")
        await self.send({"type": "done", "session_id": self.session_id})

    # ---- main loop
    async def on_json(self, m: dict) -> None:
        t = m.get("type")
        if t == "ping":
            await self.send({"type": "pong", "t": m.get("t")})
        elif t == "start":
            if self.phase not in ("idle", "done"):
                return
            snap = await asyncio.to_thread(self._snapshot)
            if snap["finished"] or snap["status"] != "active":
                await self.finish()
            elif snap["current"]:
                await self.speak(snap["current"])  # resume: ask the pending question again
            else:
                await self.respond(None, [], 0.0, "start")
        elif t == "playback" and m.get("state") == "ended":
            await self.on_playback_ended(int(m.get("utterance", -1)))
        elif t == "barge_in":
            await self.barge_in()
        elif t == "answer_meta":
            self.meta = {"mouth": m.get("mouth"), "gaze": m.get("gaze")}
            self.meta_event.set()
        elif t == "text_answer" and self.phase in ("listening", "speaking"):
            self.utterance += 1
            if self.tts_task:
                self.tts_task.cancel()
            self.audio_q.put(("discard",))
            self.turn_id += 1
            await self.respond(str(m.get("text", ""))[:8000], [], 0.0, "typed")
        elif t == "skip" and self.phase == "listening":
            self.audio_q.put(("discard",))
            self.turn_id += 1
            await self.respond("", [], 0.0, "skipped")
        elif t == "repeat" and self.phase in ("listening", "speaking"):
            snap = await asyncio.to_thread(self._snapshot)
            if snap["current"]:
                self.audio_q.put(("discard",))
                self.turn_id += 1
                await self.speak(snap["current"])
        elif t == "end":
            self.audio_q.put(("discard",))
            await self.finish()

    def on_audio(self, data: bytes) -> None:
        if self.phase == "listening" and self.stt_provider is not None:
            self.turn_audio.extend(data)
            self.audio_q.put(data)
        elif self.phase == "speaking" and self.audio_out:
            self.recent_in.extend(data)
            if len(self.recent_in) > 80000:  # 2.5 s at 16 kHz PCM16
                del self.recent_in[: len(self.recent_in) - 80000]
            self.audio_q.put(("gate_audio", data))

    async def run(self) -> None:
        old = _live.get(self.session_id)
        if old is not None:
            await old.close(4409, "replaced by a newer connection")
        _live[self.session_id] = self
        snap = await asyncio.to_thread(self._snapshot)
        self.language = snap["language"]
        self.voice = (
            snap["persona"]["kokoro_voice"]
            if self.tts_provider and self.tts_provider.name == "kokoro"
            else (
                snap["persona"]["polly_voice"]
                if self.tts_provider and self.tts_provider.name == "polly"
                else None
            )
        )
        self.worker.start()
        self.consumer = asyncio.create_task(self._consume())
        caps = runtime.capabilities()
        await self.send(
            {
                "type": "ready",
                "session": snap,
                "providers": {k: caps[k] for k in ("llm", "llm_model", "llm_fallback", "stt", "tts")},
                "audio": {"input_sample_rate": 16000, "format": "pcm16le"},
            }
        )
        if self.stt_provider is None:
            await self.send(
                {
                    "type": "error",
                    "code": "stt_unavailable",
                    "message": "Speech recognition is not available; type your answers.",
                    "recoverable": True,
                }
            )
        try:
            while True:
                msg = await self.ws.receive()
                if msg["type"] == "websocket.disconnect":
                    break
                if msg.get("bytes"):
                    self.on_audio(msg["bytes"])
                elif msg.get("text"):
                    try:
                        m = json.loads(msg["text"])
                        if not isinstance(m, dict):
                            raise TypeError
                    except (ValueError, TypeError):
                        await self.send({"type": "error", "code": "bad_message", "recoverable": True})
                        continue
                    # Commands that may wait on the LLM run in order on a consumer task, so this loop
                    # keeps reading mic frames and pings (a blocked reader stalls the socket).
                    if m.get("type") in SLOW_COMMANDS:
                        self.cmd_q.put_nowait(m)
                    else:
                        await self.on_json(m)
        except WebSocketDisconnect:
            pass
        finally:
            await self.cleanup()

    async def _consume(self) -> None:
        while True:
            m = await self.cmd_q.get()
            try:
                await self.on_json(m)
            except Exception:
                log.exception("command failed: %s", m.get("type"))

    async def close(self, code: int, reason: str) -> None:
        with contextlib.suppress(Exception):
            await self.ws.close(code=code, reason=reason)
        await self.cleanup()

    async def cleanup(self) -> None:
        if _live.get(self.session_id) is self:
            del _live[self.session_id]
        self.utterance += 1
        for task in (self.tts_task, self.playback_timer, self.consumer):
            if task:
                task.cancel()
        self.audio_q.put(_STOP)


@router.websocket("/ws/interview")
async def ws_interview(ws: WebSocket, ticket: str = "") -> None:
    await ws.accept()
    try:
        session_id, user_id = check_ticket(ticket)
    except PermissionError as e:
        await ws.close(code=4401, reason=str(e))
        return
    conv = Conversation(ws, session_id, user_id)
    await conv.run()
