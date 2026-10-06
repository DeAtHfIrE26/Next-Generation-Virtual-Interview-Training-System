"""Shared interview logic for the realtime WebSocket and the REST endpoints.

``advance()`` is the single place an answer is recorded and the next interviewer turn produced:
it computes the observable per-answer signals (delivery from word timings, E4 lip-sync
verification, E3 voice match, E5 gaze), runs the interviewer agent, meters its LLM calls and
persists the state.
"""

from __future__ import annotations

import logging
import os
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from datetime import UTC, datetime
from typing import Any

import numpy as np
from interview_core.adapters import factory
from interview_core.agent.interviewer import LOCAL_PROVIDERS, Answer, InterviewerAgent, llm_timeout_s
from interview_core.agent.json_provider import ChatJSONProvider
from interview_core.agent.state import AgentState, Turn
from interview_core.crypto import EncryptedBlob, decrypt_template
from interview_core.delivery import compute as delivery_metrics
from interview_core.gaze import summarise as gaze_summary
from interview_core.lipsync import verify_av_sync
from interview_core.nlp.evaluator import evaluate
from interview_core.nlp.structured import StructuredLLM
from interview_core.report import build_report
from interview_core.security import EventType, IntegrityMonitor, Mode, PolicyConfig
from interview_core.speech.asr import Word
from interview_core.voice import VoiceSession, VoiceVerifier
from sqlalchemy.orm import Session

from interview_api import metering, runtime
from interview_api.db import session_scope
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


# ----------------------------------------------------------------------------- pre-planning

# The interview plan (blueprint) is generated as soon as the session is created, while the
# candidate is still on the device check, so the first question only waits for one LLM call.
# advance() waits for a plan that is still in flight instead of generating a second one.
PREPLAN = os.getenv("PREPLAN", "1") != "0"
_plan_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="preplan")
_planning: dict[str, Future] = {}
_planning_lock = threading.Lock()


def preplan(session_id: str, user_id: str) -> None:
    if not PREPLAN:
        return

    def run() -> None:
        with session_scope() as db:
            s = db.get(InterviewSession, session_id)
            user = db.get(User, user_id)
            if s is None or user is None:
                return
            st = AgentState.from_dict(dict(s.state))
            if st.blueprint is not None or st.turns:
                return
            agent = agent_for(db, user, s)
            db.commit()  # no transaction held during the LLM call
            agent.plan(st)
            db.refresh(s)
            cur = AgentState.from_dict(dict(s.state))
            if cur.blueprint is None and not cur.turns:  # nobody planned meanwhile
                s.state = st.to_dict()
            metering.record_agent_attempts(db, user, s.id, "interviewer", agent.attempts)
            db.commit()

    def done(_f: Future) -> None:
        with _planning_lock:
            _planning.pop(session_id, None)

    with _planning_lock:
        fut = _plan_pool.submit(run)
        _planning[session_id] = fut
    fut.add_done_callback(done)


def wait_for_plan(session_id: str, timeout: float = 600.0) -> bool:
    """Block until an in-flight pre-plan for this session finishes. True if one was waited for."""
    with _planning_lock:
        fut = _planning.get(session_id)
    if fut is None:
        return False
    try:
        fut.result(timeout=timeout)
    except Exception:  # a failed pre-plan is redone by the turn itself
        log.exception("pre-planning failed session=%s", session_id)
    return True


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
    if wait_for_plan(s.id):
        db.refresh(s)
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


# ----------------------------------------------------------------------------- answer scoring (E6, E9)

# How long ``finish`` waits for the LLM rubric scoring before returning the report. A hosted model
# finishes well inside it; a slow (CPU-hosted) model keeps scoring in the background and the report
# is rebuilt when it is done (D21).
FINISH_EVAL_BUDGET_S = float(os.getenv("FINISH_EVAL_BUDGET_S", "20"))
_eval_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="evaluate")
_scoring: dict[str, Future] = {}
_scoring_lock = threading.Lock()


def _to_score(st: AgentState) -> list[Turn]:
    return [t for t in st.turns if t.answer is not None and (t.evaluation or {}).get("method") != "llm"]


def _score(chain: list, st: AgentState, turns: list[Turn]) -> tuple[dict[int, dict], list]:
    """LLM rubric evaluation of each answer, with the interviewer's timeout for this provider chain.
    One at a time on a local model (parallel requests only slow each other down there)."""

    def one(t: Turn):
        llm = StructuredLLM(ChatJSONProvider(chain[0]), timeout_s=llm_timeout_s(chain))
        q = {"question": t.say, "category": t.action, "difficulty": t.difficulty}
        ev = evaluate(llm, q, t.answer or "", role=st.params.role, seniority=st.params.seniority)
        return t.index, ev.to_dict(), llm.log

    workers = 1 if chain[0].name in LOCAL_PROVIDERS else 4
    with ThreadPoolExecutor(max_workers=workers) as pool:
        out = list(pool.map(one, turns))
    return {i: ev for i, ev, _ in out if ev["method"] == "llm"}, [r for _, _, log in out for r in log]


def _offline(st: AgentState) -> None:
    """Fast offline scoring (labelled "offline scoring" in the report) for answers not scored yet."""
    offline = StructuredLLM(None)
    for t in st.turns:
        if t.answer is not None and t.evaluation is None:
            q = {"question": t.say, "category": t.action, "difficulty": t.difficulty}
            t.evaluation = evaluate(
                offline, q, t.answer, role=st.params.role, seniority=st.params.seniority
            ).to_dict()


def _report(s: InterviewSession, st: AgentState, scoring: str) -> dict:
    report = build_report(
        st.to_dict(),
        per_answer=list(s.signals),
        integrity=s.integrity.get("episodes", {}),
        mode=s.mode,
        code_results=[c for c in s.code_results if c.get("final")],
    )
    report["scoring"] = scoring  # "pending" while LLM scoring continues in the background
    return report


def _apply(db: Session, user: User, s: InterviewSession, results: dict[int, dict], calls: list) -> AgentState:
    st = AgentState.from_dict(dict(s.state))
    for t in st.turns:
        if t.index in results:
            t.evaluation = results[t.index]
    metering.record_llm_calls(db, user, s.id, calls)
    return st


def _finish_later(session_id: str, user_id: str, fut: Future) -> None:
    def done(f: Future) -> None:
        with _scoring_lock:
            _scoring.pop(session_id, None)
        try:
            results, calls = f.result()
        except Exception:
            log.exception("background scoring failed session=%s", session_id)
            results, calls = {}, []
        with session_scope() as db:
            s = db.get(InterviewSession, session_id)
            user = db.get(User, user_id)
            if s is None or user is None:
                return
            st = _apply(db, user, s, results, calls)
            s.state = st.to_dict()
            s.report = _report(s, st, "complete")
            db.commit()
            log.info("background scoring done session=%s llm_scored=%d", session_id, len(results))

    with _scoring_lock:
        _scoring[session_id] = fut
    fut.add_done_callback(done)


def finish(db: Session, user: User, s: InterviewSession) -> dict:
    """Score the answers and build the report (E9). Answers are scored by the LLM rubric when a
    provider is available; whatever is not scored within ``FINISH_EVAL_BUDGET_S`` is shown with
    offline scoring for now and rescored in the background."""
    st = AgentState.from_dict(dict(s.state))
    todo = _to_score(st)
    chain = [] if metering.over_cap(db, user, s) else list(runtime.llm_chain())
    db.commit()  # no transaction held during the LLM calls
    fut = _eval_pool.submit(_score, chain, st, todo) if chain and todo else None
    results, calls, pending = {}, [], False
    if fut is not None:
        try:
            results, calls = fut.result(timeout=FINISH_EVAL_BUDGET_S)
        except FutureTimeout:
            pending = True
    db.refresh(s)
    if s.report is not None:  # a concurrent finish got there first
        return s.report
    st = _apply(db, user, s, results, calls)
    _offline(st)
    s.state = st.to_dict()
    s.report = _report(s, st, "pending" if pending else "complete")
    if s.status == "active":
        s.status = "finished"
    s.finished_at = datetime.now(UTC)
    db.commit()
    if pending:
        _finish_later(s.id, user.id, fut)
    return s.report


def resume_scoring(db: Session, s: InterviewSession) -> None:
    """Restart background scoring that a process restart interrupted (the report still says pending)."""
    if (s.report or {}).get("scoring") != "pending":
        return
    with _scoring_lock:
        if s.id in _scoring:
            return
    user = db.get(User, s.user_id)
    st = AgentState.from_dict(dict(s.state))
    chain = [] if user is None or metering.over_cap(db, user, s) else list(runtime.llm_chain())
    if not chain or user is None:
        s.report = {**s.report, "scoring": "complete"}
        db.commit()
        return
    _finish_later(s.id, user.id, _eval_pool.submit(_score, chain, st, _to_score(st)))
