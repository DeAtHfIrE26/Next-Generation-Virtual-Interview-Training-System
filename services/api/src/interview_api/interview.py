"""Shared interview logic for the realtime WebSocket and the REST endpoints.

``advance()`` is the single place an answer is recorded and the next interviewer turn produced:
it computes the observable per-answer signals (delivery from word timings, E4 lip-sync
verification, E3 voice match, E5 gaze), runs the interviewer agent, meters its LLM calls and
persists the state.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

import numpy as np
from interview_core.adapters import factory
from interview_core.agent.interviewer import Answer, InterviewerAgent, llm_timeout_s
from interview_core.agent.state import AgentState, Turn
from interview_core.crypto import EncryptedBlob, decrypt_template
from interview_core.delivery import compute as delivery_metrics
from interview_core.gaze import summarise as gaze_summary
from interview_core.lipsync import verify_av_sync
from interview_core.security import EventType, IntegrityMonitor, Mode, PolicyConfig
from interview_core.speech.asr import Word
from interview_core.voice import VoiceSession, VoiceVerifier
from sqlalchemy.orm import Session

from interview_api import metering, runtime
from interview_api.models import BiometricTemplate, InterviewSession, User
from interview_api.routers.consent import latest

log = logging.getLogger("interview_api.interview")


# ----------------------------------------------------------------------------- public views


def public_params(st: AgentState) -> dict:
    p = st.params
    return {
        "role": p.role,
        "seniority": p.seniority,
        "company": p.company,
        "interview_type": p.interview_type,
        "round": p.round,
        "difficulty": p.difficulty,
        "language": p.language,
        "duration_minutes": p.duration_minutes,
        "persona": p.persona,
        "skills": p.skills,
    }


def public_blueprint(st: AgentState) -> dict | None:
    bp = st.blueprint
    if bp is None:
        return None
    cov = st.coverage()
    return {
        "summary": bp.summary,
        "emergency": bp.emergency,
        "competencies": [
            {"id": c.id, "name": c.name, "minutes": c.minutes, "questions": cov.get(c.id, {}).get("turns", 0)}
            for c in bp.competencies
        ],
    }


def public_turn(st: AgentState, t: Turn | None) -> dict | None:
    if t is None:
        return None
    comp = st.blueprint.by_id(t.competency) if st.blueprint else None
    return {
        "index": t.index,
        "action": t.action,
        "competency": comp.name if comp else t.competency,
        "difficulty": t.difficulty,
        "say": t.say,
        "emergency": t.emergency,
        "answered": t.answer is not None,
        "answer": t.answer,
    }


# ----------------------------------------------------------------------------- integrity helpers


def monitor(s: InterviewSession) -> IntegrityMonitor:
    return IntegrityMonitor.from_state(s.integrity.get("_monitor"), PolicyConfig(mode=Mode(s.mode)))


def save_monitor(s: InterviewSession, m: IntegrityMonitor) -> None:
    s.integrity = {**s.integrity, "_monitor": m.to_state(), "episodes": m.summary()}


def notice_out(n) -> dict:
    return {"event": n.event.value, "message": n.message, "episode": n.episode, "end_session": n.end_session}


def template(db: Session, user: User, kind: str):
    row = db.query(BiometricTemplate).filter_by(user_id=user.id, kind=kind).first()
    if row is None:
        return None
    exp = row.expires_at if row.expires_at.tzinfo else row.expires_at.replace(tzinfo=UTC)
    if exp < datetime.now(UTC):
        return None
    return decrypt_template(
        EncryptedBlob.from_json(row.blob), user.id, kind, row.model_id, runtime.key_provider()
    )


# ----------------------------------------------------------------------------- the turn


def agent_for(db: Session, user: User, s: InterviewSession) -> InterviewerAgent:
    """The configured provider chain, or none once the session's hard cost cap is reached (the agent
    then uses flagged emergency questions; the UI shows the badge)."""
    chain = [] if metering.over_cap(db, user, s) else list(runtime.llm_chain())
    return InterviewerAgent(chain, timeout_s=llm_timeout_s(chain))


def answer_signals(
    db: Session,
    user: User,
    s: InterviewSession,
    index: int,
    words: list[Word],
    audio: np.ndarray | None,
    sr: int,
    mouth: dict | None,
    gaze: list | None,
    transcript_source: str,
) -> tuple[dict[str, Any], list[dict]]:
    signals: dict[str, Any] = {"transcript_source": transcript_source}
    notices: list[dict] = []
    mon = monitor(s)
    t_now = float(index)
    signals["delivery"] = delivery_metrics(words).to_dict() if words else {"measured": False}
    if gaze:
        g = gaze_summary(gaze)
        signals["gaze"] = {**g, "measured": g["coverage"] > 0}
    if (
        audio is not None
        and mouth
        and len(mouth.get("times", [])) == len(mouth.get("values", []))
        and len(mouth["times"]) > 10
    ):
        vals = np.array([np.nan if v is None else v for v in mouth["values"]], dtype=float)
        r = verify_av_sync(audio, sr, np.asarray(mouth["times"], float), vals)  # E4
        signals["lipsync"] = {"measured": r.decision != "inconclusive", **r.__dict__}
        if r.decision == "mismatch" and (n := mon.single(EventType.LIPSYNC_MISMATCH, t_now)):
            notices.append(notice_out(n))
    voice_tpl = template(db, user, "voice") if latest(db, user.id).get("biometric_voice") else None
    spk = runtime.speaker_embedder()
    if audio is not None and voice_tpl is not None and spk is not None:  # E3
        chk = VoiceSession(VoiceVerifier(spk, factory.voice_threshold()), voice_tpl).check(audio, sr)
        signals["voice"] = {"measured": chk.score is not None, "status": chk.status.value, "score": chk.score}
        if chk.status.value == "mismatch" and (n := mon.single(EventType.VOICE_MISMATCH, t_now)):
            notices.append(notice_out(n))
    else:
        signals["voice"] = {
            "measured": False,
            "status": "not_enrolled" if voice_tpl is None else "not_configured",
        }
    save_monitor(s, mon)
    if any(n["end_session"] for n in notices):
        s.status = "ended_by_policy"
    return signals, notices


def advance(
    db: Session,
    user: User,
    s: InterviewSession,
    text: str | None,
    *,
    words: list[Word],
    seconds: float,
    source: str,
    audio: np.ndarray | None = None,
    sr: int = 16000,
    mouth: dict | None = None,
    gaze: list | None = None,
) -> tuple[Turn | None, AgentState, list[dict]]:
    """Record ``text`` as the answer to the pending question (None = no answer yet, e.g. start),
    then produce the next interviewer turn. Persists everything before returning."""
    st = AgentState.from_dict(dict(s.state))
    pending = st.awaiting_answer
    answer = None
    notices: list[dict] = []
    if pending is not None and text is not None:
        answer = Answer(text, [w.__dict__ for w in words], seconds)
        signals, notices = answer_signals(db, user, s, pending.index, words, audio, sr, mouth, gaze, source)
        sig = list(s.signals) + [None] * (pending.index + 1 - len(s.signals))
        sig[pending.index] = signals
        s.signals = sig
    agent = agent_for(db, user, s)
    active = s.status == "active"
    # End the transaction before the (slow) LLM call so other requests on this session, such as
    # integrity events, are never blocked behind it; the turn is written in a new transaction.
    db.commit()
    if active:
        turn = agent.next_turn(st, answer)
    else:  # ended by policy: record the answer, no further questions
        if answer is not None and pending is not None:
            pending.answer, pending.answer_seconds = answer.text, answer.seconds
        st.finished, turn = True, None
    metering.record_agent_attempts(db, user, s.id, "interviewer", agent.attempts)
    if turn is not None and turn.emergency:
        metering.record_emergency(db, s.id)
    s.state = st.to_dict()
    db.commit()
    return turn, st, notices
