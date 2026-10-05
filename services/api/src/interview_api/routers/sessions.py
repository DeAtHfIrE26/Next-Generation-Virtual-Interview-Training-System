"""Interview sessions: the end-to-end flow that ties every patent element together.

E1 receive role/resume -> E6 adaptive questions -> answer with E3 voice match, E4 lip-sync,
E5 gaze, E7 integrity events, E2 periodic face checks -> E6 evaluation -> E8 coding
challenge -> E9 report.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal

import numpy as np
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from interview_core.adapters import factory
from interview_core.avatar import from_polly
from interview_core.codeexec import grade_submission, load_challenges, pick_challenge
from interview_core.crypto import EncryptedBlob, decrypt_template
from interview_core.delivery import compute as delivery_metrics
from interview_core.face import FaceVerifier
from interview_core.gaze import summarise as gaze_summary
from interview_core.lipsync import verify_av_sync
from interview_core.nlp.interviewer import Interviewer, InterviewState
from interview_core.nlp.resume import parse_resume_pdf
from interview_core.nlp.roles import is_technical
from interview_core.nlp.structured import StructuredLLM
from interview_core.report import build_report
from interview_core.security import EventType, IntegrityMonitor, Mode, PolicyConfig
from interview_core.speech.asr import Word
from interview_core.voice import VoiceSession, VoiceVerifier
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from interview_api import metering, runtime
from interview_api.db import get_db
from interview_api.media import decode_image, decode_wav
from interview_api.models import BiometricTemplate, InterviewSession, SessionEvent, User
from interview_api.ratelimit import limiter
from interview_api.routers.consent import latest, require
from interview_api.security import current_user

router = APIRouter(prefix="/sessions", tags=["sessions"])
turn_limit = limiter("turn", capacity=60, per_seconds=60)
code_limit = limiter("code", capacity=20, per_seconds=60)
MAX_RESUME_BYTES = 2 * 1024 * 1024


# ----------------------------------------------------------------------------- helpers


def _own(db: Session, user: User, session_id: str) -> InterviewSession:
    s = db.get(InterviewSession, session_id)
    if s is None or s.user_id != user.id:
        raise HTTPException(404, "session not found")
    return s


def _active(s: InterviewSession) -> None:
    if s.status != "active":
        raise HTTPException(409, f"session is {s.status.replace('_', ' ')}")


def _llm(db: Session, user: User, s: InterviewSession) -> StructuredLLM:
    """The configured LLM unless the session's hard cost cap is reached (then offline bank)."""
    if metering.over_cap(db, user, s):
        return StructuredLLM(None)
    return StructuredLLM(runtime.llm_provider())


def _monitor(s: InterviewSession) -> IntegrityMonitor:
    return IntegrityMonitor.from_state(s.integrity.get("_monitor"), PolicyConfig(mode=Mode(s.mode)))


def _save_monitor(s: InterviewSession, m: IntegrityMonitor) -> None:
    s.integrity = {**s.integrity, "_monitor": m.to_state(), "episodes": m.summary()}


def _notice_out(n) -> dict:
    return {"event": n.event.value, "message": n.message, "episode": n.episode, "end_session": n.end_session}


def _template(db: Session, user: User, kind: str):
    row = db.query(BiometricTemplate).filter_by(user_id=user.id, kind=kind).first()
    if row is None:
        return None
    exp = row.expires_at if row.expires_at.tzinfo else row.expires_at.replace(tzinfo=UTC)
    if exp < datetime.now(UTC):
        return None
    return decrypt_template(
        EncryptedBlob.from_json(row.blob), user.id, kind, row.model_id, runtime.key_provider()
    )


def _public_question(turn, index: int) -> dict:
    q = turn.question
    return {
        "index": index,
        "question": q["question"],
        "category": q["category"],
        "difficulty": q["difficulty"],
        "source": turn.source,
        "follow_up": turn.source == "follow_up",
    }


# ----------------------------------------------------------------------------- create / list


@router.post("", status_code=201)
async def create_session(
    role: str = Form(min_length=2, max_length=120),
    seniority: Literal["intern", "junior", "mid", "senior", "lead"] = Form("mid"),
    job_description: str = Form("", max_length=6000),
    mode: Literal["coaching", "proctored"] = Form("coaching"),
    length: int = Form(8, ge=3, le=15),
    resume: UploadFile | None = File(None),
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
) -> dict:
    require(db, user, "data_processing")
    limits = metering.plan_limits(user.plan)
    if metering.sessions_this_month(db, user) >= limits.sessions_per_month:
        raise HTTPException(402, f"your plan allows {limits.sessions_per_month} sessions per month")
    context, profile = "", {}
    if resume is not None:
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
        profile = {
            "skills": p.skills[:30],
            "years_experience": p.years_experience,
            "sections": sorted(p.sections),
            "truncated": p.truncated,
        }
    s = InterviewSession(
        user_id=user.id,
        role=role.strip(),
        seniority=seniority,
        mode=mode,
        state={},
        signals=[],
        integrity={"resume_profile": profile},
        code_results=[],
    )
    db.add(s)
    db.flush()
    s.state = InterviewState.start(s.id, s.role, seniority, context, job_description, length).to_dict()
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
    return [
        {
            "id": r.id,
            "role": r.role,
            "seniority": r.seniority,
            "mode": r.mode,
            "status": r.status,
            "created_at": r.created_at.isoformat(),
            "overall": (r.report or {}).get("summary", {}).get("overall"),
            "label": (r.report or {}).get("summary", {}).get("label"),
        }
        for r in rows
    ]


@router.get("/{session_id}")
def get_session(session_id: str, user: User = Depends(current_user), db: Session = Depends(get_db)) -> dict:
    s = _own(db, user, session_id)
    st = InterviewState.from_dict(dict(s.state))
    current = st.turns[-1] if st.turns and st.turns[-1].answer is None else None
    return {
        "id": s.id,
        "role": s.role,
        "seniority": s.seniority,
        "mode": s.mode,
        "status": s.status,
        "answered": sum(t.answer is not None for t in st.turns),
        "planned": len(st.plan),
        "difficulty": st.difficulty,
        "current": _public_question(current, len(st.turns) - 1) if current else None,
        "integrity": s.integrity.get("episodes", {}),
        "capabilities": runtime.capabilities(),
    }


# ----------------------------------------------------------------------------- questions


@router.post("/{session_id}/next", dependencies=[Depends(turn_limit)])
def next_question(session_id: str, user: User = Depends(current_user), db: Session = Depends(get_db)) -> dict:
    s = _own(db, user, session_id)
    _active(s)
    st = InterviewState.from_dict(dict(s.state))
    llm = _llm(db, user, s)
    turn = Interviewer(llm).next_question(st)
    metering.record_llm_calls(db, user, s.id, llm.log)
    if turn is None:
        s.state = st.to_dict()
        db.commit()
        return {"done": True}
    index = len(st.turns) - 1
    out = _public_question(turn, index)
    # E8: attach one coding challenge to the first technical question for technical roles.
    if (
        turn.question["category"] == "technical"
        and is_technical(st.family)
        and not any(c.get("attached") for c in s.code_results)
    ):
        ch = pick_challenge(st.family, st.difficulty, s.id)
        if ch is not None:
            s.code_results = [*s.code_results, {"attached": True, "challenge_id": ch.id, "turn": index}]
            out["challenge"] = ch.public()
    tts = runtime.tts_provider()
    if tts is not None and metering.plan_limits(user.plan).server_tts:
        import base64

        speech = tts.synthesize(turn.question["question"])
        metering.record(db, user, s.id, "tts_characters", tts.name, speech.characters)
        out["speech"] = {
            "audio_b64": base64.b64encode(speech.audio).decode(),
            "mime": speech.mime,
            "visemes": from_polly(speech.visemes),
        }
    s.state = st.to_dict()
    db.commit()
    return out


# ----------------------------------------------------------------------------- answers


class WordIn(BaseModel):
    word: str = Field(max_length=60)
    start: float
    end: float


class MouthSeries(BaseModel):
    times: list[float] = Field(max_length=20000)
    values: list[float | None] = Field(max_length=20000)


class AnswerIn(BaseModel):
    transcript: str = Field(default="", max_length=8000)
    words: list[WordIn] | None = Field(default=None, max_length=5000)
    audio_wav: str | None = None  # base64 16-bit PCM WAV of the answer (for server ASR, E3, E4)
    mouth: MouthSeries | None = None  # E4: aperture series, seconds from audio start
    gaze_samples: list[tuple[float, bool | None]] | None = Field(default=None, max_length=20000)  # E5
    duration_s: float | None = None


@router.post("/{session_id}/answer", dependencies=[Depends(turn_limit)])
def submit_answer(
    session_id: str, body: AnswerIn, user: User = Depends(current_user), db: Session = Depends(get_db)
) -> dict:
    s = _own(db, user, session_id)
    _active(s)
    st = InterviewState.from_dict(dict(s.state))
    if not st.turns or st.turns[-1].answer is not None:
        raise HTTPException(409, "no question is waiting for an answer; call /next")
    index = len(st.turns) - 1
    signals: dict[str, Any] = {}
    notices: list[dict] = []
    monitor = _monitor(s)
    t_now = float(index)

    audio = sr = None
    if body.audio_wav:
        audio, sr = decode_wav(body.audio_wav)
    transcript, words = body.transcript.strip(), [Word(w.word, w.start, w.end) for w in body.words or []]
    asr = runtime.asr_provider()
    if audio is not None and asr is not None:
        tr = asr.transcribe(audio, sr)
        metering.record(db, user, s.id, "asr_seconds", asr.name, tr.audio_seconds)
        transcript, words = tr.text, tr.words
        signals["transcript_source"] = asr.name
    else:
        signals["transcript_source"] = "client"

    signals["delivery"] = delivery_metrics(words).to_dict() if words else {"measured": False}

    if body.gaze_samples:
        g = gaze_summary(body.gaze_samples)
        signals["gaze"] = {**g, "measured": g["coverage"] > 0}

    if audio is not None and body.mouth and len(body.mouth.times) == len(body.mouth.values):
        vals = np.array([np.nan if v is None else v for v in body.mouth.values], dtype=float)
        r = verify_av_sync(audio, sr, np.asarray(body.mouth.times, float), vals)
        signals["lipsync"] = {"measured": r.decision != "inconclusive", **r.__dict__}
        if r.decision == "mismatch" and (n := monitor.single(EventType.LIPSYNC_MISMATCH, t_now)):
            notices.append(_notice_out(n))

    voice_tpl = _template(db, user, "voice") if latest(db, user.id).get("biometric_voice") else None
    spk = runtime.speaker_embedder()
    if audio is not None and voice_tpl is not None and spk is not None:
        chk = VoiceSession(VoiceVerifier(spk, factory.voice_threshold()), voice_tpl).check(audio, sr)
        signals["voice"] = {"measured": chk.score is not None, "status": chk.status.value, "score": chk.score}
        if chk.status.value == "mismatch" and (n := monitor.single(EventType.VOICE_MISMATCH, t_now)):
            notices.append(_notice_out(n))
    else:
        signals["voice"] = {
            "measured": False,
            "status": "not_enrolled" if voice_tpl is None else "not_configured",
        }

    llm = _llm(db, user, s)
    ev = Interviewer(llm).submit_answer(st, transcript)
    metering.record_llm_calls(db, user, s.id, llm.log)
    sig = list(s.signals) + [None] * (index + 1 - len(s.signals))
    sig[index] = signals
    s.signals, s.state = sig, st.to_dict()
    _save_monitor(s, monitor)
    if any(n["end_session"] for n in notices):
        s.status = "ended_by_policy"
    db.commit()
    return {
        "index": index,
        "evaluation": ev.to_dict(),
        "signals": signals,
        "notices": notices,
        "difficulty": st.difficulty,
        "status": s.status,
        "finished": st.finished,
    }


# ----------------------------------------------------------------------------- integrity


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
    monitor = _monitor(s)
    notices = []
    for o in sorted(body.observations, key=lambda o: o.t):
        n = monitor.observe(EventType(o.type), o.present, o.t)
        if n:
            notices.append(_notice_out(n))
            db.add(SessionEvent(session_id=s.id, t=o.t, type=o.type, payload={"episode": n.episode}))
    _save_monitor(s, monitor)
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
    tpl = _template(db, user, "face") if latest(db, user.id).get("biometric_face") else None
    if emb is None or tpl is None:
        return {"status": "not_configured" if emb is None else "not_enrolled"}
    res = FaceVerifier(emb, thr).verify(tpl, decode_image(body.image))
    monitor = _monitor(s)
    status = "uncalibrated" if res.accepted is None else ("match" if res.accepted else "mismatch")
    n = monitor.observe(EventType.FACE_MISMATCH, status == "mismatch", body.t)
    _save_monitor(s, monitor)
    if n and n.end_session:
        s.status = "ended_by_policy"
    db.commit()
    return {"status": status, "score": round(res.score, 4), "notice": _notice_out(n) if n else None}


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
    if s.report is None:
        report = build_report(
            dict(s.state),
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
    return s.report
