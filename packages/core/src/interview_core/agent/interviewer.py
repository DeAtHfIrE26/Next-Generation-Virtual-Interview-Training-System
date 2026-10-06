"""The interviewer agent (patent element E6, upgraded).

Every question is generated live by an LLM, conditioned on the full parameter set (role,
seniority, company and its style, JD, resume, skills to probe, interview type, round, difficulty,
language, duration and time remaining) and the full history of answers. The agent works from an
explicit plan: a blueprint of competencies and time allocation built at session start, and
deterministic coverage bookkeeping updated after every answer.

There is no question bank. When generation fails, the chain is: retry the primary provider once
(transient errors) -> one repair attempt showing the model its validation errors -> the alternate
provider -> an LLM-free emergency question that is flagged (``Turn.emergency``, a WARNING log line
and an audit event) so the UI can show it and tests can fail on it.

Code enforces what must hold regardless of model quality:
- difficulty moves with performance (direction rules, at most one step per turn);
- follow-ups and challenges must quote the candidate's last answer verbatim (a loosely remembered
  quote is replaced by the real words it overlaps, or the reply is rejected);
- no repeated or rephrased questions (content-word Jaccard below 0.6);
- the interview wraps up and closes on time.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any

from interview_core.agent import prompts
from interview_core.agent.llm import ChatLLM
from interview_core.agent.personas import persona
from interview_core.agent.state import AgentState, Blueprint, Competency, Turn
from interview_core.nlp.heuristics import content_words
from interview_core.nlp.providers.base import PermanentLLMError, RefusalError, TransientLLMError
from interview_core.nlp.structured import wrap_untrusted

log = logging.getLogger("interview_core.agent")

ACTIONS = ("open", "follow_up", "challenge", "new_topic", "revisit", "wrap_up", "close")
DUPLICATE_JACCARD = 0.6
WRAP_UP_REMAINING_S = 75.0
MAX_SAY_CHARS = 700
_PLAN = re.compile(r"<plan>(.*?)</plan>", re.S)
_SAY = re.compile(r"<say>(.*?)(?:</say>|$)", re.S)
_ASKS = re.compile(r"\?|\b(tell me|walk me|describe|explain|talk me|give me|share|take me|go ahead)\b", re.I)


class AgentError(RuntimeError):
    pass


@dataclass
class Answer:
    text: str
    words: list[dict[str, Any]] = field(default_factory=list)
    seconds: float = 0.0


@dataclass
class Attempt:
    provider: str
    model: str
    ok: bool
    ms: float
    error: str = ""
    input_tokens: int = 0
    output_tokens: int = 0


# ----------------------------------------------------------------------------- helpers


def _norm_tokens(s: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", s.lower())


def quote_matches(quote: str, answer: str) -> bool:
    """True when ``quote`` is (nearly) verbatim in ``answer``: exact normalised substring, or at
    least 80% of its words appear in order-insensitive form (tolerates punctuation and ASR casing)."""
    q, a = _norm_tokens(quote), _norm_tokens(answer)
    if not q or not a:
        return False
    if " ".join(q) in " ".join(a):
        return True
    if len(q) < 3:
        return False
    aset = set(a)
    return sum(w in aset for w in q) / len(q) >= 0.8


def ground_quote(quote: str, answer: str) -> str | None:
    """Replace a loosely remembered quote with the real words it refers to.

    Scans ``answer`` with a window of 1.5x the quote's length for the span sharing the most words
    with ``quote``, trimmed to its first and last shared word. Returns that span, verbatim from the
    answer, when it contains at least half of the quote's words (and at least three); otherwise
    None. The stored anchor is therefore always the candidate's actual words, never a paraphrase."""
    q = _norm_tokens(quote)
    words = answer.split()
    if len(q) < 3 or not words:
        return None
    norm = [" ".join(_norm_tokens(w)) for w in words]
    qset = set(q)
    n = min(len(words), -(-len(q) * 3 // 2))
    best, span = 0, (0, 0)
    for i in range(len(words) - n + 1):
        idx = [j for j in range(i, i + n) if norm[j] and norm[j] in qset]
        hits = len({norm[j] for j in idx})
        if hits > best:
            best, span = hits, (idx[0], idx[-1])
    if best >= 3 and best / len(qset) >= 0.5:
        return " ".join(words[span[0] : span[1] + 1]).strip(" ,;:.")
    return None


def turn_schema(ids: list[str], forced: str | None, needs_last_answer: bool) -> dict[str, Any]:
    """JSON schema for one interviewer turn (used for grammar-constrained decoding)."""
    actions = [forced] if forced else [a for a in ACTIONS if a != "open"]
    props: dict[str, Any] = {}
    if needs_last_answer:
        props["last_answer"] = {
            "type": "object",
            "properties": {
                "score": {"type": "integer", "enum": [1, 2, 3, 4, 5]},
                "strengths": {"type": "string"},
                "gaps": {"type": "string"},
                "vague": {"type": "boolean"},
            },
            "required": ["score", "strengths", "gaps", "vague"],
            "additionalProperties": False,
        }
    props |= {
        "action": {"type": "string", "enum": actions},
        "competency": {"type": "string", "enum": ids},
        "difficulty": {"type": "integer", "enum": [1, 2, 3, 4, 5]},
        "anchor_quote": {"type": "string"},
        "reason": {"type": "string"},
        "say": {"type": "string"},
    }
    return {"type": "object", "properties": props, "required": list(props), "additionalProperties": False}


BLUEPRINT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "competencies": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "name": {"type": "string"},
                    "why": {"type": "string"},
                    "weight": {"type": "number"},
                    "minutes": {"type": "number"},
                    "signals": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["id", "name", "why", "weight", "minutes", "signals"],
                "additionalProperties": False,
            },
        },
        "opening": {"type": "string"},
        "style_notes": {"type": "string"},
        "start_difficulty": {"type": "integer", "enum": [1, 2, 3, 4, 5]},
    },
    "required": ["summary", "competencies", "opening", "style_notes", "start_difficulty"],
    "additionalProperties": False,
}

JSON_FORMAT_NOTE = (
    "\n\nOutput format for this conversation: reply with a single JSON object containing the plan "
    'fields and a "say" field holding exactly what you say out loud (instead of <plan> and <say> tags).'
)


def jaccard(a: str, b: str) -> float:
    wa, wb = content_words(a), content_words(b)
    return len(wa & wb) / len(wa | wb) if wa and wb else 0.0


def is_repeat(a: str, b: str) -> bool:
    """Near-duplicate questions: Jaccard >= 0.6, or (for questions of 4+ content words) an overlap
    coefficient >= 0.75, which catches a short question re-asked inside a longer one."""
    wa, wb = content_words(a), content_words(b)
    if not wa or not wb:
        return False
    if len(wa & wb) / len(wa | wb) >= DUPLICATE_JACCARD:
        return True
    small = min(len(wa), len(wb))
    return small >= 4 and len(wa & wb) / small >= 0.75


def parse_reply(text: str) -> tuple[dict[str, Any] | None, str, list[str]]:
    """Accepts the tagged format (``<plan>{json}</plan><say>...</say>``) or, from providers using
    schema-constrained output, one JSON object holding the plan fields plus ``say``."""
    errors: list[str] = []
    plan = None
    stripped = text.strip()
    if stripped.startswith("{") and "<plan>" not in stripped:
        try:
            obj = json.loads(stripped)
        except json.JSONDecodeError as e:
            return None, "", [f"reply is not valid JSON ({e.msg})"]
        if not isinstance(obj, dict):
            return None, "", ["reply must be a JSON object"]
        say = " ".join(str(obj.pop("say", "") or "").split())
        return obj, say, ([] if say else ["missing say"])
    m = _PLAN.search(text)
    if not m:
        errors.append("missing <plan>...</plan>")
    else:
        raw = m.group(1).strip().removeprefix("```json").removesuffix("```").strip()
        try:
            plan = json.loads(raw)
            if not isinstance(plan, dict):
                errors.append("plan must be a JSON object")
                plan = None
        except json.JSONDecodeError as e:
            errors.append(f"plan is not valid JSON ({e.msg})")
    s = _SAY.search(text)
    say = " ".join(s.group(1).split()) if s else ""
    if not say:
        errors.append("missing <say>...</say>")
    return plan, say, errors


def _extract_json(text: str) -> dict[str, Any]:
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t)
    start, end = t.find("{"), t.rfind("}")
    if start < 0 or end < start:
        raise ValueError("no JSON object in reply")
    return json.loads(t[start : end + 1])


# ----------------------------------------------------------------------------- agent


class InterviewerAgent:
    def __init__(self, chain: list[ChatLLM], *, timeout_s: float = 30.0, transient_retries: int = 1):
        self.chain = chain
        self.timeout_s = timeout_s
        self.transient_retries = transient_retries
        self.attempts: list[Attempt] = []

    # ---- low-level call with retry and provider fail-over
    def _call(
        self,
        system: str,
        messages: list[dict[str, str]],
        *,
        max_tokens: int,
        effort: str,
        validate,
        schema: dict[str, Any] | None = None,
        json_note: str = "",
    ):
        """Return (result, provider_name) or (None, None) when every provider failed."""
        for llm in self.chain:
            use_schema = schema is not None and getattr(llm, "supports_schema", False)
            sys_prompt = system + json_note if use_schema else system
            corrections: list[str] = []
            repaired = False
            tries = 0
            while True:
                tries += 1
                t0 = time.monotonic()
                msgs = messages
                if corrections:
                    msgs = [
                        *messages,
                        {
                            "role": "user",
                            "content": "Your previous reply was rejected: "
                            + "; ".join(corrections)
                            + ". Reply again in the required format.",
                        },
                    ]
                try:
                    text = "".join(
                        llm.stream(
                            sys_prompt,
                            msgs,
                            max_tokens=max_tokens,
                            timeout_s=self.timeout_s,
                            effort=effort,
                            **({"schema": schema} if use_schema else {}),
                        )
                    )
                except TransientLLMError as e:
                    self._record(llm, False, t0, str(e))
                    if tries <= self.transient_retries:
                        time.sleep(0.4 * tries)
                        continue
                    break
                except (PermanentLLMError, RefusalError) as e:
                    self._record(llm, False, t0, str(e))
                    break
                result, errors = validate(text)
                self._record(llm, not errors, t0, "; ".join(errors))
                if not errors:
                    return result, llm.name
                if repaired:
                    break
                repaired, corrections = True, errors
        return None, None

    def _record(self, llm: ChatLLM, ok: bool, t0: float, error: str) -> None:
        u = getattr(llm, "last_usage", None)
        self.attempts.append(
            Attempt(
                llm.name,
                llm.model,
                ok,
                round((time.monotonic() - t0) * 1000, 1),
                error[:300],
                getattr(u, "input_tokens", 0),
                getattr(u, "output_tokens", 0),
            )
        )
        if not ok:
            log.warning("llm attempt failed provider=%s model=%s error=%s", llm.name, llm.model, error[:200])

    # ---- blueprint
    def plan(self, st: AgentState) -> Blueprint:
        p = st.params

        def validate(text: str):
            try:
                d = _extract_json(text)
            except (ValueError, json.JSONDecodeError) as e:
                return None, [f"reply is not one JSON object ({e})"]
            errs = []
            comps = d.get("competencies")
            if not isinstance(comps, list) or not 2 <= len(comps) <= 7:
                errs.append("competencies must be a list of 3 to 6 items")
                return None, errs
            seen = set()
            parsed = []
            for i, c in enumerate(comps):
                if not isinstance(c, dict) or not str(c.get("name", "")).strip():
                    errs.append(f"competency {i + 1} needs a name")
                    continue
                cid = str(c.get("id") or f"c{i + 1}")
                if cid in seen:
                    cid = f"c{i + 1}"
                seen.add(cid)
                try:
                    minutes = float(c.get("minutes", 0))
                    weight = float(c.get("weight", 0))
                except (TypeError, ValueError):
                    errs.append(f"competency {cid} minutes/weight must be numbers")
                    continue
                parsed.append(
                    Competency(
                        cid,
                        str(c["name"])[:80],
                        str(c.get("why", ""))[:300],
                        weight,
                        minutes,
                        [str(s)[:160] for s in c.get("signals", [])][:5],
                    )
                )
            total = sum(c.minutes for c in parsed)
            if total > p.duration_minutes + 1:
                errs.append(
                    f"minutes add up to {total:g}, more than the {p.duration_minutes}-minute duration"
                )
            missing = [
                s for s in p.skills if not any(s.lower() in (c.name + " " + c.why).lower() for c in parsed)
            ]
            if missing and len(missing) == len(p.skills):
                errs.append("include the skills to probe: " + ", ".join(p.skills))
            if errs:
                return None, errs
            try:
                sd = int(d.get("start_difficulty", p.start_difficulty))
            except (TypeError, ValueError):
                sd = p.start_difficulty
            bp = Blueprint(
                str(d.get("summary", ""))[:500],
                parsed,
                str(d.get("opening", ""))[:300],
                str(d.get("style_notes", ""))[:400],
                max(1, min(5, sd)),
            )
            return bp, []

        brief = prompts.session_brief(p)
        bp, provider = self._call(
            prompts.PLANNER_SYSTEM,
            [{"role": "user", "content": brief + "\n\nWrite the blueprint now."}],
            max_tokens=1600,
            effort="medium",
            validate=validate,
            schema=BLUEPRINT_SCHEMA,
        )
        if bp is None:
            bp = emergency_blueprint(st)
            log.warning(
                "EMERGENCY blueprint used session=%s (no LLM produced a valid blueprint)", st.session_id
            )
            st.log("emergency_blueprint")
        else:
            bp.provider = provider or ""
        st.blueprint = bp
        if p.difficulty == "auto":
            st.difficulty = bp.start_difficulty
        return bp

    # ---- one interviewer turn
    def next_turn(
        self, st: AgentState, answer: Answer | None = None, now: float | None = None
    ) -> Turn | None:
        """Record ``answer`` for the pending question, then generate the next interviewer turn.
        Returns None once the interview has closed."""
        now = now or time.time()
        if st.blueprint is None:
            self.plan(st)
        pending = st.awaiting_answer
        if pending is not None:
            if answer is None:
                return pending  # idempotent: still waiting for this answer
            pending.answer = answer.text.strip()
            pending.answer_words = answer.words
            pending.answer_seconds = answer.seconds
            pending.answered_at = now
        if st.finished:
            return None
        if not st.started_at:
            st.started_at = now

        last = st.turns[-1] if st.turns else None
        forced = None
        if last is None:
            forced = "open"
        elif last.action == "wrap_up":
            forced = "close"
        elif st.remaining_s(now) <= WRAP_UP_REMAINING_S:
            forced = "wrap_up"

        system = prompts.interviewer_system(st)
        messages = self._history(st)
        messages.append({"role": "user", "content": prompts.turn_state(st, now, forced, [])})
        ids = {c.id for c in st.blueprint.competencies}
        last_answer = last.answer if last is not None else None

        def validate(text: str):
            plan, say, errs = parse_reply(text)
            if plan is None:
                return None, errs
            action = plan.get("action")
            if action not in ACTIONS:
                errs.append(f"action must be one of {ACTIONS}")
            if forced and action != forced:
                errs.append(f"action must be {forced} now")
            if not forced and action == "open":
                errs.append("open is only for the first turn")
            if action != "close" and plan.get("competency") not in ids:
                errs.append(f"competency must be one of {sorted(ids)}")
            try:
                diff = int(plan.get("difficulty"))
                if not 1 <= diff <= 5:
                    raise ValueError
            except (TypeError, ValueError):
                errs.append("difficulty must be an integer 1-5")
            la = plan.get("last_answer")
            if last_answer is not None and not isinstance(la, dict):
                errs.append("last_answer must assess the candidate's previous answer")
            elif isinstance(la, dict):
                try:
                    if not 1 <= int(la.get("score")) <= 5:
                        raise ValueError
                except (TypeError, ValueError):
                    errs.append("last_answer.score must be an integer 1-5")
            quote = str(plan.get("anchor_quote") or "").strip()
            if action in ("follow_up", "challenge"):
                if not quote:
                    errs.append(f"{action} needs an anchor_quote from the last answer")
                elif " ".join(_norm_tokens(quote)) not in " ".join(_norm_tokens(last_answer or "")):
                    real = ground_quote(quote, last_answer or "")
                    if real is None:
                        errs.append("anchor_quote must be copied verbatim from the candidate's last answer")
                    else:
                        plan["anchor_quote"] = real
                        plan["_grounded_from"] = quote
            if len(say) > MAX_SAY_CHARS:
                errs.append(f"say must be under {MAX_SAY_CHARS} characters")
            if re.search(r"(^|\s)([-*•]|\d+\.)\s", say) or "**" in say:
                errs.append("say must be plain spoken text without lists or markdown")
            if action not in ("close",) and say and not _ASKS.search(say):
                errs.append("say must ask the candidate a question")
            for prev in st.turns:
                if say and is_repeat(say, prev.say):
                    errs.append(f"repeats an earlier question: {prev.say[:80]!r}")
                    break
            return (plan, say), errs

        t0 = time.monotonic()
        res, provider = self._call(
            system,
            messages,
            max_tokens=900,
            effort="low",
            validate=validate,
            schema=turn_schema(sorted(ids), forced, last_answer is not None),
            json_note=JSON_FORMAT_NOTE,
        )
        gen_ms = round((time.monotonic() - t0) * 1000, 1)
        if res is None:
            turn = emergency_turn(st, forced, now)
            log.warning(
                "EMERGENCY question used session=%s turn=%d (all LLM providers failed)",
                st.session_id,
                turn.index,
            )
            st.log("emergency_question", turn=turn.index)
        else:
            plan, say = res
            turn = self._apply(st, plan, say, provider or "", now)
            st.log("turn", turn=turn.index, provider=provider, ms=gen_ms, action=turn.action)
        st.turns.append(turn)
        if turn.action == "close":
            st.finished = True
        return turn

    def _history(self, st: AgentState) -> list[dict[str, str]]:
        msgs: list[dict[str, str]] = [
            {"role": "user", "content": "The candidate has joined. Begin when instructed."}
        ]
        for t in st.turns:
            msgs.append({"role": "assistant", "content": f"<say>{t.say}</say>"})
            if t.answer is not None:
                text = t.answer if t.answer else "(no answer: the candidate stayed silent)"
                msgs.append({"role": "user", "content": "Candidate's answer:\n" + wrap_untrusted(text)})
        return msgs

    def _apply(self, st: AgentState, plan: dict[str, Any], say: str, provider: str, now: float) -> Turn:
        corrections: list[str] = []
        prev_diff = st.difficulty
        la = plan.get("last_answer") if isinstance(plan.get("last_answer"), dict) else None
        score = int(la["score"]) if la else None
        if st.turns and la:
            st.turns[-1].score = score
            st.turns[-1].assessment = {k: la.get(k) for k in ("strengths", "gaps", "vague")}
        if plan.get("_grounded_from"):
            corrections.append(
                f"anchor_quote grounded to the candidate's words (model wrote {plan['_grounded_from'][:80]!r})"
            )
        wanted = int(plan["difficulty"])
        diff = enforce_difficulty(prev_diff, wanted, score, st.params.difficulty)
        if diff != wanted:
            corrections.append(f"difficulty {wanted}->{diff} (score {score}, previous {prev_diff})")
        st.difficulty = diff
        action = plan["action"]
        comp = plan.get("competency") or (
            st.turns[-1].competency if st.turns else st.blueprint.competencies[0].id
        )
        turn = Turn(
            index=len(st.turns),
            action=action,
            competency=comp,
            difficulty=diff,
            say=say,
            anchor_quote=str(plan.get("anchor_quote") or "")[:300],
            reason=str(plan.get("reason") or "")[:300],
            provider=provider,
            asked_at=now,
            corrections=corrections,
        )
        for c in corrections:
            st.log("correction", turn=turn.index, detail=c)
        return turn


def enforce_difficulty(prev: int, wanted: int, score: int | None, requested: str) -> int:
    """Difficulty tracks performance: at most one step per turn; a strong answer (4-5) never lowers it
    and a 5 raises it; a weak answer (1-2) never raises it and a 1 lowers it. With a fixed requested
    difficulty the level stays within one step of the request."""
    d = max(prev - 1, min(prev + 1, wanted))
    if score is not None:
        if score >= 4:
            d = max(d, prev)
        if score == 5:
            d = max(d, min(5, prev + 1))
        if score <= 2:
            d = min(d, prev)
        if score == 1:
            d = min(d, max(1, prev - 1))
    if requested != "auto":
        r = int(requested)
        d = max(r - 1, min(r + 1, d))
    return max(1, min(5, d))


# ----------------------------------------------------------------------------- emergency (LLM-free)

_GENERIC = {
    "technical": ["Technical fundamentals", "Problem solving", "Depth on a past project"],
    "behavioral": ["Ownership", "Collaboration", "Handling conflict and setbacks"],
    "system_design": ["Requirements and scoping", "Architecture", "Trade-offs and scaling"],
    "hr": ["Motivation", "Values and ways of working", "Career goals"],
    "case": ["Structuring the problem", "Analysis", "Recommendation"],
    "mixed": ["Background and motivation", "Role-specific skills", "Teamwork"],
}


def emergency_blueprint(st: AgentState) -> Blueprint:
    p = st.params
    names = (p.skills[:3] or []) + [n for n in _GENERIC[p.interview_type] if n not in p.skills]
    names = names[:3]
    minutes = max(2.0, (p.duration_minutes - 2) / len(names))
    comps = [
        Competency(
            f"c{i + 1}",
            n,
            "Emergency blueprint: no LLM was available.",
            round(1 / len(names), 2),
            round(minutes, 1),
        )
        for i, n in enumerate(names)
    ]
    return Blueprint(
        "Emergency blueprint built without an LLM.",
        comps,
        "Greet and ask for a background overview.",
        "",
        p.start_difficulty,
        emergency=True,
    )


def emergency_turn(st: AgentState, forced: str | None, now: float) -> Turn:
    who = persona(st.params.persona).name
    idx = len(st.turns)
    bp = st.blueprint
    assert bp is not None
    if forced == "open":
        say = f"Hello, I'm {who}, and I'll be your interviewer today. To start, could you give me a short overview of your background and what draws you to this {st.params.role} role?"
        action, comp = "open", bp.competencies[0].id
    elif forced == "close":
        say = "That's all the time we have. Thank you for your answers today, and good luck."
        action, comp = "close", st.turns[-1].competency if st.turns else bp.competencies[0].id
    elif forced == "wrap_up":
        say = "We're almost out of time. Is there anything you'd like to ask me about the role or the team?"
        action, comp = "wrap_up", bp.competencies[-1].id
    else:
        cov = st.coverage()
        target = min(bp.competencies, key=lambda c: (cov[c.id]["turns"], -c.weight))
        say = f"Let's talk about {target.name.lower()}. Can you walk me through a specific situation where this mattered, what you did yourself, and what the outcome was?"
        action, comp = "new_topic", target.id
    return Turn(idx, action, comp, st.difficulty, say, provider="emergency", emergency=True, asked_at=now)
