"""Interview sessions: creation, state, typed-answer fallback, integrity, code and the report.

The live spoken interview runs over the realtime WebSocket (``interview_api.live``). These REST
endpoints cover everything else and a typed-answer path that uses the same interviewer agent, so
an interview never depends on the microphone working.

E1 receive role/JD/resume -> E6 interviewer agent (live LLM questions) -> answers with E3 voice
match, E4 lip-sync, E5 gaze, E7 integrity events, E2 periodic face checks -> E6 evaluation ->
E8 coding challenge -> E9 report.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from typing import Literal

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from interview_core.adapters import factory
from interview_core.agent.json_provider import ChatJSONProvider
from interview_core.agent.personas import PERSONAS
from interview_core.agent.state import (
    INTERVIEW_TYPES,
    LANGUAGES,
    ROUNDS,
    SENIORITIES,
    AgentState,
    InterviewParams,
)
from interview_core.codeexec import grade_submission, load_challenges, pick_challenge
from interview_core.face import FaceVerifier
from interview_core.nlp.evaluator import evaluate
from interview_core.nlp.resume import parse_resume_pdf
from interview_core.nlp.roles import is_technical, role_family
from interview_core.nlp.structured import StructuredLLM
from interview_core.realtime import stt
from interview_core.report import build_report
from interview_core.security import EventType
from interview_core.speech.asr import Word
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from interview_api import interview, metering, runtime
from interview_api.db import get_db
from interview_api.media import decode_image, decode_wav
from interview_api.models import InterviewSession, SessionEvent, User
from interview_api.ratelimit import limiter
from interview_api.routers.consent import latest, require
from interview_api.security import current_user

router = APIRouter(prefix="/sessions", tags=["sessions"])
turn_limit = limiter("turn", capacity=60, per_seconds=60)
code_limit = limiter("code", capacity=20, per_seconds=60)
MAX_RESUME_BYTES = 2 * 1024 * 1024


def _own(db: Session, user: User, session_id: str) -> InterviewSession:
    s = db.get(InterviewSession, session_id)
    if s is None or s.user_id != user.id:
        raise HTTPException(404, "session not found")
    return s


def _active(s: InterviewSession) -> None:
    if s.status != "active":
        raise HTTPException(409, f"session is {s.status.replace('_', ' ')}")


@router.get("/options")
def options() -> dict:
    """Everything the set-up screen offers, so the client never hardcodes it."""
    return {
        "seniorities": list(SENIORITIES),
        "interview_types": list(INTERVIEW_TYPES),
        "rounds": list(ROUNDS),
        "languages": LANGUAGES,
        "personas": [p.public() for p in PERSONAS.values()],
        "capabilities": runtime.capabilities(),
    }


@router.post("", status_code=201)
async def create_session(
    role: str = Form(min_length=2, max_length=120),
    seniority: Literal["intern", "junior", "mid", "senior", "lead", "principal"] = Form("mid"),
    company: str = Form("", max_length=120),
    company_style: str = Form("", max_length=400),
    job_description: str = Form("", max_length=6000),
    skills: str = Form("", max_length=600),  # comma-separated
    interview_type: Literal["technical", "behavioral", "system_design", "hr", "case", "mixed"] = Form(
        "mixed"
    ),
    round: Literal["screening", "technical", "onsite", "final", "hr"] = Form("technical"),
    difficulty: Literal["auto", "1", "2", "3", "4", "5"] = Form("auto"),
    language: str = Form("en", max_length=5),
    duration_minutes: int = Form(20, ge=5, le=60),
    persona: str = Form("maya", max_length=20),
    mode: Literal["coaching", "proctored"] = Form("coaching"),
    resume: UploadFile | None = File(None),
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
) -> dict:
    require(db, user, "data_processing")
    limits = metering.plan_limits(user.plan)
    if metering.sessions_this_month(db, user) >= limits.sessions_per_month:
        raise HTTPException(402, f"your plan allows {limits.sessions_per_month} sessions per month")
    if language not in LANGUAGES:
        raise HTTPException(422, f"language must be one of {sorted(LANGUAGES)}")
    context, profile = "", {}
    if resume is not None and resume.filename:
        data = await resume.read(MAX_RESUME_BYTES + 1)
        if len(data) > MAX_RESUME_BYTES:
            raise HTTPException(413, "resume must be under 2 MB")
        if not data.startswith(b"%PDF"):
            raise HTTPException(422, "resume must be a PDF")
        try:
            p = parse_resume_pdf(data)
        except Exception as e:  # malformed PDFs raise many types
            raise HTTPException(422, "could not read this PDF") from e
        context = p.context_for_llm()
        profile = {"skills": p.skills[:30], "years_experience": p.years_experience, "truncated": p.truncated}
    params = InterviewParams(
        role=role,
        seniority=seniority,
        company=company,
        company_style=company_style,
        job_description=job_description,
        resume_context=context,
        skills=[s for s in skills.split(",")],
        interview_type=interview_type,
        round=round,
        difficulty=difficulty,
        language=language,
        duration_minutes=duration_minutes,
        persona=persona if persona in PERSONAS else "maya",
    )
    s = InterviewSession(
        user_id=user.id,
        role=params.role,
        seniority=seniority,
        mode=mode,
        state={},
        signals=[],
        integrity={"resume_profile": profile},
        code_results=[],
    )
    db.add(s)
    db.flush()
    s.state = AgentState.new(s.id, params).to_dict()
    # E8: technical interviews of technical roles get one coding exercise alongside the conversation.
    if interview_type in ("technical", "mixed", "system_design") and is_technical(role_family(role)):
        ch = pick_challenge(role_family(role), params.start_difficulty, s.id)
        if ch is not None:
            s.code_results = [{"attached": True, "challenge_id": ch.id}]
    db.commit()
    return {
        "id": s.id,
        "resume_profile": profile,
        "capabilities": runtime.capabilities(),
        "consents": latest(db, user.id),
    }


@router.get("")
def list_sessions(user: User = Depends(current_user), db: Session = Depends(get_db)) -> list[dict]:
    rows = (
        db.query(InterviewSession)
        .filter_by(user_id=user.id)
        .order_by(InterviewSession.created_at.desc())
        .limit(100)
    )
    out = []
    for r in rows:
        params = (r.state or {}).get("params", {})
        summary = (r.report or {}).get("summary", {})
        out.append(
            {
                "id": r.id,
                "role": r.role,
                "company": params.get("company") or None,
                "interview_type": params.get("interview_type"),
                "seniority": r.seniority,
                "mode": r.mode,
                "status": r.status,
                "created_at": r.created_at.isoformat(),
                "overall": summary.get("overall"),
                "label": summary.get("label"),
                "dimensions": summary.get("dimensions"),
            }
        )
    return out


@router.get("/{session_id}")
def get_session(session_id: str, user: User = Depends(current_user), db: Session = Depends(get_db)) -> dict:
    s = _own(db, user, session_id)
    st = AgentState.from_dict(dict(s.state))
    attached = next((c for c in s.code_results if c.get("attached")), None)
    challenge = None
    if attached:
        challenge = next((c.public() for c in load_challenges() if c.id == attached["challenge_id"]), None)
    return {
        "id": s.id,
        "status": s.status,
        "mode": s.mode,
        "params": interview.public_params(st),
        "persona": PERSONAS.get(st.params.persona, PERSONAS["maya"]).public(),
        "blueprint": interview.public_blueprint(st),
        "turns": [interview.public_turn(st, t) for t in st.turns],
        "current": interview.public_turn(st, st.awaiting_answer) if st.awaiting_answer else None,
        "finished": st.finished,
        "remaining_s": st.remaining_s() if st.started_at else st.params.duration_minutes * 60,
        "challenge": challenge,
        "integrity": s.integrity.get("episodes", {}),
        "capabilities": runtime.capabilities(),
    }


class WordIn(BaseModel):
    word: str = Field(max_length=60)
    start: float
    end: float


class MouthSeries(BaseModel):
    times: list[float] = Field(max_length=20000)
    values: list[float | None] = Field(max_length=20000)


class TurnIn(BaseModel):
    """``text`` None and no audio = start (no answer yet). With ``audio_wav`` and no text the server
    transcribes the recording (upload fallback for networks without WebSockets)."""

    text: str | None = Field(default=None, max_length=8000)
    audio_wav: str | None = None  # base64 16-bit PCM WAV of the answer
    words: list[WordIn] | None = Field(default=None, max_length=5000)
    mouth: MouthSeries | None = None  # E4: aperture series, seconds from audio start
    gaze_samples: list[tuple[float, bool | None]] | None = Field(default=None, max_length=20000)  # E5


@router.post("/{session_id}/turn", dependencies=[Depends(turn_limit)])
def turn(
    session_id: str, body: TurnIn, user: User = Depends(current_user), db: Session = Depends(get_db)
) -> dict:
    """Record the answer (typed or uploaded recording) and return the next interviewer turn."""
    s = _own(db, user, session_id)
    _active(s)
    audio = sr = None
    if body.audio_wav:
        audio, sr = decode_wav(body.audio_wav)
    text, words, source = body.text, [Word(w.word, w.start, w.end) for w in body.words or []], "typed"
    if text is None and audio is not None:
        prov = runtime.stt_provider()
        if prov is None:
            raise HTTPException(503, "speech recognition is not available; send text")
        tr = stt.transcribe(prov, audio, sr)
        metering.record(db, user, s.id, "asr_seconds", prov.name, tr.audio_seconds)
        text, words, source = tr.text, tr.words, prov.name
    elif audio is not None:
        source = "client"
    seconds = float(len(audio) / sr) if audio is not None else 0.0
    nxt, st, notices = interview.advance(
        db,
        user,
        s,
        text,
        words=words,
        seconds=seconds,
        source=source,
        audio=audio,
        sr=sr or 16000,
        mouth=body.mouth.model_dump() if body.mouth else None,
        gaze=body.gaze_samples,
    )
    idx = (nxt.index - 1) if nxt else len(st.turns) - 1
    return {
        "turn": interview.public_turn(st, nxt) if nxt else None,
        "signals": s.signals[idx] if text is not None and 0 <= idx < len(s.signals) else None,
        "answered": st.turns[nxt.index - 1].answer if nxt and nxt.index > 0 else None,
        "finished": st.finished,
        "status": s.status,
        "notices": notices,
        "difficulty": st.difficulty,
    }


# ----------------------------------------------------------------------------- integrity (E7)


class Observation(BaseModel):
    t: float
    type: Literal["phone", "second_person", "no_face", "background_noise", "second_speaker"]
    present: bool


class EventsIn(BaseModel):
    observations: list[Observation] = Field(max_length=2000)


@router.post("/{session_id}/events")
def post_events(
    session_id: str, body: EventsIn, user: User = Depends(current_user), db: Session = Depends(get_db)
) -> dict:
    s = _own(db, user, session_id)
    _active(s)
    monitor = interview.monitor(s)
    notices = []
    for o in sorted(body.observations, key=lambda o: o.t):
        n = monitor.observe(EventType(o.type), o.present, o.t)
        if n:
            notices.append(interview.notice_out(n))
            db.add(SessionEvent(session_id=s.id, t=o.t, type=o.type, payload={"episode": n.episode}))
    interview.save_monitor(s, monitor)
    if any(n["end_session"] for n in notices):
        s.status = "ended_by_policy"
    db.commit()
    return {"notices": notices, "status": s.status, "episodes": monitor.summary()}


class FaceCheckIn(BaseModel):
    image: str
    t: float = 0.0


@router.post("/{session_id}/face-check", dependencies=[Depends(turn_limit)])
def face_check(
    session_id: str, body: FaceCheckIn, user: User = Depends(current_user), db: Session = Depends(get_db)
) -> dict:
    """E2 continuous verification: periodic face crop checked against the enrolled template."""
    s = _own(db, user, session_id)
    _active(s)
    emb, thr = runtime.face_embedder(), factory.face_threshold()
    tpl = interview.template(db, user, "face") if latest(db, user.id).get("biometric_face") else None
    if emb is None or tpl is None:
        return {"status": "not_configured" if emb is None else "not_enrolled"}
    res = FaceVerifier(emb, thr).verify(tpl, decode_image(body.image))
    monitor = interview.monitor(s)
    status = "uncalibrated" if res.accepted is None else ("match" if res.accepted else "mismatch")
    n = monitor.observe(EventType.FACE_MISMATCH, status == "mismatch", body.t)
    interview.save_monitor(s, monitor)
    if n and n.end_session:
        s.status = "ended_by_policy"
    db.commit()
    return {"status": status, "score": round(res.score, 4), "notice": interview.notice_out(n) if n else None}


# ----------------------------------------------------------------------------- code (E8)


class CodeIn(BaseModel):
    language: Literal["python", "javascript", "java", "cpp", "sql"]
    source: str = Field(max_length=64000)
    final: bool = False


@router.post("/{session_id}/code", dependencies=[Depends(code_limit)])
def run_code(
    session_id: str, body: CodeIn, user: User = Depends(current_user), db: Session = Depends(get_db)
) -> dict:
    s = _own(db, user, session_id)
    _active(s)
    attached = next((c for c in s.code_results if c.get("attached")), None)
    if attached is None:
        raise HTTPException(409, "no coding challenge in this session")
    ch = next(c for c in load_challenges() if c.id == attached["challenge_id"])
    result = grade_submission(ch, body.language, body.source, runtime.judge0())
    metering.record(db, user, s.id, "code_runs", result.executor or "none", len(result.outcomes))
    out = {
        "challenge_id": ch.id,
        "language": body.language,
        "passed": result.passed,
        "total": result.total,
        "summary": result.summary(),
        "error": result.error,
        "outcomes": [o.__dict__ for o in result.outcomes],
    }
    if body.final:
        s.code_results = [*s.code_results, {**out, "final": True, "source_chars": len(body.source)}]
    db.commit()
    return out


# ----------------------------------------------------------------------------- finish (E9)


@router.post("/{session_id}/finish")
def finish(session_id: str, user: User = Depends(current_user), db: Session = Depends(get_db)) -> dict:
    s = _own(db, user, session_id)
    if s.report is not None:
        return s.report
    st = AgentState.from_dict(dict(s.state))
    pending = [t for t in st.turns if t.answer is not None and t.evaluation is None]
    chain = list(runtime.llm_chain()) if not metering.over_cap(db, user, s) else []
    llm = StructuredLLM(ChatJSONProvider(chain[0]) if chain else None)

    def run(t):
        q = {"question": t.say, "category": t.action, "difficulty": t.difficulty}
        return t, evaluate(llm, q, t.answer or "", role=st.params.role, seniority=st.params.seniority)

    with ThreadPoolExecutor(max_workers=4) as pool:
        for t, ev in pool.map(run, pending):
            t.evaluation = ev.to_dict()
    metering.record_llm_calls(db, user, s.id, llm.log)
    s.state = st.to_dict()
    report = build_report(
        s.state,
        per_answer=list(s.signals),
        integrity=s.integrity.get("episodes", {}),
        mode=s.mode,
        code_results=[c for c in s.code_results if c.get("final")],
    )
    s.report = report
    if s.status == "active":
        s.status = "finished"
    s.finished_at = datetime.now(UTC)
    db.commit()
    return report
