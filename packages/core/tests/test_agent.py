"""Interviewer agent: validation, repair, provider fail-over, difficulty rules, timing, persistence.

The LLM here is a scripted test double (test code only). Real-LLM behaviour is proven by
eval/agent_mock_interviews.py, which writes transcripts to docs/evidence/questions/.
"""

import json

import pytest
from interview_core.agent.interviewer import (
    Answer,
    InterviewerAgent,
    enforce_difficulty,
    ground_quote,
    parse_reply,
    quote_matches,
    turn_schema,
)
from interview_core.agent.llm import Usage
from interview_core.agent.state import AgentState, InterviewParams
from interview_core.nlp.providers.base import PermanentLLMError, TransientLLMError

BLUEPRINT = {
    "summary": "Assesses backend depth and ownership.",
    "competencies": [
        {
            "id": "c1",
            "name": "API design",
            "why": "JD asks for REST APIs",
            "weight": 0.4,
            "minutes": 6,
            "signals": ["versioning"],
        },
        {
            "id": "c2",
            "name": "Databases",
            "why": "resume: Postgres",
            "weight": 0.3,
            "minutes": 5,
            "signals": [],
        },
        {"id": "c3", "name": "Ownership", "why": "senior role", "weight": 0.3, "minutes": 5, "signals": []},
    ],
    "opening": "Greet, then ask about a recent API.",
    "style_notes": "Direct.",
    "start_difficulty": 3,
}


def reply(action, comp="c1", diff=3, say="Tell me about an API you designed recently?", quote="", last=None):
    plan = {
        "last_answer": last,
        "action": action,
        "competency": comp,
        "difficulty": diff,
        "anchor_quote": quote,
        "reason": "r",
    }
    return f"<plan>{json.dumps(plan)}</plan>\n<say>{say}</say>"


class Scripted:
    """Yields scripted replies (or raises scripted exceptions) in order."""

    def __init__(self, name, script):
        self.name, self.model, self.script, self.calls = name, "test-model", list(script), []
        self.last_usage = Usage(10, 5)

    def stream(self, system, messages, *, max_tokens=1200, timeout_s=30.0, effort="low"):
        self.calls.append({"system": system, "messages": messages, "effort": effort})
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        yield from (item[i : i + 7] for i in range(0, len(item), 7))  # stream in small chunks


def state(**kw):
    kw = {"duration_minutes": 20, "skills": ["Postgres"], **kw}
    return AgentState.new("s1", InterviewParams(role="Backend Engineer", **kw))


def test_full_interview_flow_follow_up_difficulty_and_close():
    llm = Scripted(
        "primary",
        [
            json.dumps(BLUEPRINT),
            reply("open", say="Hi, I'm Maya. Tell me about an API you designed recently?"),
            reply(
                "follow_up",
                diff=4,
                say="You said you added cursor pagination. Why cursors rather than offsets?",
                quote="added cursor pagination",
                last={"score": 4, "strengths": "specific", "gaps": "", "vague": False},
            ),
            reply(
                "new_topic",
                comp="c2",
                diff=3,
                say="How would you find and fix a slow Postgres query?",
                last={"score": 2, "strengths": "", "gaps": "no trade-offs", "vague": True},
            ),
        ],
    )
    ag = InterviewerAgent([llm])
    st = state()
    t0 = ag.next_turn(st, now=1000.0)
    assert st.blueprint and [c.id for c in st.blueprint.competencies] == ["c1", "c2", "c3"]
    assert t0.action == "open" and not t0.emergency and t0.provider == "primary"
    t1 = ag.next_turn(
        st, Answer("We added cursor pagination and versioned the endpoints.", seconds=30), now=1060.0
    )
    assert t1.action == "follow_up" and t1.anchor_quote == "added cursor pagination"
    assert st.turns[0].score == 4 and t1.difficulty == 4
    # the candidate's answer reached the model inside untrusted-input tags
    last_msgs = llm.calls[-1]["messages"]
    assert any("<candidate_input>" in m["content"] and "cursor pagination" in m["content"] for m in last_msgs)
    t2 = ag.next_turn(st, Answer("I'd add an index I guess.", seconds=8), now=1100.0)
    assert t2.difficulty == 3  # weak answer (score 2): difficulty did not rise
    assert st.coverage()["c1"]["scores"] == [4, 2]


def test_invalid_anchor_quote_is_repaired_once():
    llm = Scripted(
        "primary",
        [
            json.dumps(BLUEPRINT),
            reply("open"),
            reply(
                "follow_up",
                say="You mentioned Kafka. Why Kafka?",
                quote="we used Kafka",
                last={"score": 3, "strengths": "", "gaps": "", "vague": False},
            ),
            reply(
                "follow_up",
                say="You said the cache was stale. How did you notice?",
                quote="the cache was stale",
                last={"score": 3, "strengths": "", "gaps": "", "vague": False},
            ),
        ],
    )
    ag = InterviewerAgent([llm])
    st = state()
    ag.next_turn(st, now=0.5)
    t = ag.next_turn(st, Answer("Honestly the cache was stale for hours.", seconds=10), now=30)
    assert t.anchor_quote == "the cache was stale" and not t.emergency
    assert "verbatim" in llm.calls[-1]["messages"][-1]["content"]  # repair prompt showed the error


def test_duplicate_question_rejected():
    q = "Tell me about an API you designed recently?"
    llm = Scripted(
        "p",
        [
            json.dumps(BLUEPRINT),
            reply("open", say=q),
            reply(
                "new_topic",
                comp="c2",
                say="So tell me about an API that you designed recently?",
                last={"score": 3, "strengths": "", "gaps": "", "vague": False},
            ),
            reply(
                "new_topic",
                comp="c2",
                say="How do you choose indexes in Postgres?",
                last={"score": 3, "strengths": "", "gaps": "", "vague": False},
            ),
        ],
    )
    ag = InterviewerAgent([llm])
    st = state()
    ag.next_turn(st, now=1)
    t = ag.next_turn(st, Answer("answer", seconds=5), now=20)
    assert t.say.startswith("How do you choose indexes")


def test_transient_retry_then_fallback_provider_then_emergency():
    primary = Scripted(
        "primary", [json.dumps(BLUEPRINT), TransientLLMError("timeout"), TransientLLMError("timeout")]
    )
    backup = Scripted("backup", [reply("open", say="Hello, tell me about yourself?")])
    ag = InterviewerAgent([primary, backup])
    st = state()
    t = ag.next_turn(st, now=1)
    assert t.provider == "backup" and not t.emergency
    assert [a.ok for a in ag.attempts] == [True, False, False, True]

    dead = Scripted("dead", [PermanentLLMError("401"), PermanentLLMError("401")])
    ag2 = InterviewerAgent([dead])
    st2 = state()
    t2 = ag2.next_turn(st2, now=1)
    assert st2.blueprint.emergency and t2.emergency and t2.provider == "emergency"
    assert {e["kind"] for e in st2.events} >= {"emergency_blueprint", "emergency_question"}


def test_wrap_up_and_close_are_forced_by_time():
    short = {**BLUEPRINT, "competencies": [{**c, "minutes": 1} for c in BLUEPRINT["competencies"]]}
    llm = Scripted(
        "p",
        [
            json.dumps(short),
            reply("open"),
            reply(
                "wrap_up",
                comp="c3",
                say="We're nearly out of time. Do you have any questions for me?",
                last={"score": 3, "strengths": "", "gaps": "", "vague": False},
            ),
            reply(
                "close",
                comp="c3",
                say="Thank you for your time today. Goodbye.",
                last={"score": 3, "strengths": "", "gaps": "", "vague": False},
            ),
        ],
    )
    ag = InterviewerAgent([llm])
    st = state(duration_minutes=5)
    ag.next_turn(st, now=0.0 + 1)
    t = ag.next_turn(st, Answer("answer", seconds=200), now=1 + 250)  # 50 s left
    assert t.action == "wrap_up" and "nearly up" in llm.calls[-1]["messages"][-1]["content"]
    t2 = ag.next_turn(st, Answer("No questions.", seconds=3), now=1 + 260)
    assert t2.action == "close" and st.finished
    assert ag.next_turn(st, None, now=300) is None


@pytest.mark.parametrize(
    "prev,wanted,score,requested,expected",
    [
        (3, 5, 5, "auto", 4),  # at most one step
        (3, 3, 5, "auto", 4),  # a 5 raises
        (3, 2, 4, "auto", 3),  # a 4 never lowers
        (3, 4, 2, "auto", 3),  # a 2 never raises
        (3, 3, 1, "auto", 2),  # a 1 lowers
        (1, 1, 1, "auto", 1),  # floor
        (5, 5, 5, "auto", 5),  # ceiling
        (3, 4, None, "auto", 4),  # opening: free within one step
        (4, 5, 5, "2", 3),  # fixed requested difficulty: within one of it
    ],
)
def test_difficulty_rules(prev, wanted, score, requested, expected):
    assert enforce_difficulty(prev, wanted, score, requested) == expected


def test_quote_matching_tolerates_asr_punctuation():
    ans = "So, um, we added cursor-based pagination and versioned it."
    assert quote_matches("added cursor based pagination", ans)
    assert not quote_matches("we migrated to Kafka", ans)


def test_state_round_trips_through_json():
    llm = Scripted("p", [json.dumps(BLUEPRINT), reply("open")])
    st = state(company="Acme", interview_type="technical")
    InterviewerAgent([llm]).next_turn(st, now=5)
    st2 = AgentState.from_dict(json.loads(json.dumps(st.to_dict())))
    assert st2.blueprint.competencies[1].name == "Databases" and st2.turns[0].say == st.turns[0].say
    assert st2.params.company == "Acme"


def test_params_validation():
    with pytest.raises(ValueError):
        InterviewParams(role="x", interview_type="poetry")
    p = InterviewParams(role="SWE", duration_minutes=500, skills=[" Go ", ""])
    assert p.duration_minutes == 60 and p.skills == ["Go"]


class SchemaScripted(Scripted):
    """A provider with grammar-constrained JSON output (like Ollama or Gemini)."""

    supports_schema = True

    def stream(self, system, messages, *, max_tokens=1200, timeout_s=30.0, effort="low", schema=None):
        self.calls.append({"system": system, "messages": messages, "effort": effort, "schema": schema})
        yield self.script.pop(0)


def test_schema_providers_get_a_json_schema_and_json_replies_are_accepted():
    turn = {
        "action": "open",
        "competency": "c1",
        "difficulty": 3,
        "anchor_quote": "",
        "reason": "r",
        "say": "Hi, I'm Maya. Which API did you design most recently?",
    }
    llm = SchemaScripted("ollama", [json.dumps(BLUEPRINT), json.dumps(turn)])
    agent = InterviewerAgent([llm])
    st = state()
    t = agent.next_turn(st, now=1000.0)
    assert t is not None and not t.emergency and t.action == "open"
    assert llm.calls[0]["schema"]["required"][1] == "competencies"
    s = llm.calls[1]["schema"]
    assert s["properties"]["action"]["enum"] == ["open"]  # forced action is enforced by the grammar
    assert s["properties"]["competency"]["enum"] == ["c1", "c2", "c3"]
    assert "last_answer" not in s["properties"]  # nothing to assess before the first answer
    assert '"say" field' in llm.calls[1]["system"]


def test_turn_schema_requires_assessment_after_an_answer():
    s = turn_schema(["c1", "c2"], None, True)
    assert "open" not in s["properties"]["action"]["enum"]
    assert "last_answer" in s["required"] and s["additionalProperties"] is False


def test_parse_reply_accepts_both_formats():
    plan, say, errs = parse_reply('{"action": "close", "say": "Thanks,  goodbye."}')
    assert errs == [] and plan == {"action": "close"} and say == "Thanks, goodbye."
    plan, say, errs = parse_reply('<plan>{"action": "close"}</plan><say>Bye.</say>')
    assert errs == [] and say == "Bye."
    assert parse_reply("{not json")[2]


def test_loose_quote_is_grounded_to_the_candidates_real_words():
    answer = "So I rewrote the nightly batch job in Go and it went from six hours down to forty minutes."
    real = ground_quote("rewrote the nightly job in Golang, six hours to forty minutes", answer)
    assert real is not None and real in answer and "nightly" in real
    assert ground_quote("we migrated to Kubernetes last spring", answer) is None


def test_grounded_follow_up_stores_verbatim_anchor_and_logs_correction():
    answer = "I led the move from cron scripts to Airflow and cut failed runs by sixty percent."
    llm = Scripted(
        "p",
        [
            json.dumps(BLUEPRINT),
            reply("open", say="Hi, I'm Maya. What data pipeline work have you owned?"),
            reply(
                "follow_up",
                quote="moved from cron to Airflow and cut failures by sixty percent",
                last={"score": 3, "strengths": "", "gaps": "", "vague": False},
                say="How did you measure that sixty percent drop in failures?",
            ),
        ],
    )
    agent = InterviewerAgent([llm])
    st = state()
    agent.next_turn(st, now=1000.0)
    t = agent.next_turn(st, Answer(answer), now=1060.0)
    assert t is not None and not t.emergency and t.action == "follow_up"
    assert t.anchor_quote in answer
    assert any("grounded" in c for c in t.corrections)


def test_from_dict_does_not_mutate_its_input():
    st = state()
    st.blueprint = InterviewerAgent([Scripted("p", [json.dumps(BLUEPRINT)])]).plan(st)
    d = json.loads(json.dumps(st.to_dict()))
    AgentState.from_dict(d)
    assert AgentState.from_dict(d).blueprint.competencies[0].id == "c1"
