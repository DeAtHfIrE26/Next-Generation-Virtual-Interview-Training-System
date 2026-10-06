"""Generate full mock interviews with the REAL interviewer agent and a simulated candidate.

Usage:
  uv run python eval/agent_mock_interviews.py --out docs/evidence/questions [--only 0-4] [--shard i/n]

The interviewer is the production agent (interview_core.agent) on whatever LLM chain the
environment configures (LLM_PROVIDER, ...). The candidate is a separate LLM persona with an assigned
answer-quality profile, so we can check that difficulty follows performance. Time is simulated
(answers take as long as they would at 150 words per minute) so time-keeping logic runs without
real waiting.

Every transcript is checked and the checks are written next to it:
  1. no question overlaps the retired question bank (exact or near-duplicate),
  2. follow-ups and challenges reference the candidate's previous answer,
  3. difficulty tracks performance (after strong answers it never drops; after weak ones it never rises),
  4. no emergency (LLM-free) question was used,
  5. no question repeats an earlier one.
Nothing here is mocked: if the LLM is unavailable, the run fails loudly.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from interview_core.agent.interviewer import Answer, InterviewerAgent, is_repeat, quote_matches
from interview_core.agent.llm import ChatLLM, chain_from_env
from interview_core.agent.state import AgentState, InterviewParams
from interview_core.nlp.heuristics import content_words

ROOT = Path(__file__).resolve().parents[1]
BANK = json.loads((ROOT / "packages/core/tests/fixtures/legacy_question_bank.json").read_text())


def bank_texts() -> list[str]:
    items = BANK if isinstance(BANK, list) else BANK.get("questions", [])
    return [q.get("text") or q.get("question") or "" for q in items]


@dataclass
class Case:
    params: InterviewParams
    profile: str  # strong | average | weak | vague | mixed_up_down | improving


JD_BACKEND = "Build and operate high-throughput REST and gRPC services in Go and Python on Kubernetes; own on-call; Postgres and Kafka."
JD_DS = "Develop forecasting and experimentation (A/B testing) for a marketplace; Python, SQL, causal inference, stakeholder communication."
JD_PM = "Own the onboarding funnel for a B2B SaaS product; define metrics, run discovery with customers, work with design and engineering."
JD_SALES = "Close mid-market deals for a payments product; run discovery calls, manage a pipeline in Salesforce, negotiate contracts."
JD_NURSE = "Provide patient care in a 30-bed surgical ward; medication administration, patient education, escalation of deteriorating patients."
JD_FE = "Build accessible, performant React and TypeScript interfaces; design systems; web vitals; collaborate with designers."
RESUME_BACKEND = "5 years backend. Built payment reconciliation service in Go (Postgres, Kafka). Reduced p99 latency 40%. Led migration from monolith to services. On-call lead."
RESUME_JUNIOR = "B.Tech CSE 2025. Internship: built a Flask REST API for inventory. Projects: chat app with WebSockets, LeetCode 300 problems."
RESUME_DS = "M.Sc Statistics. 3 years data scientist at an e-commerce company: demand forecasting (Prophet, LightGBM), ran 40+ A/B tests."

CASES: list[Case] = [
    Case(
        InterviewParams(
            role="Backend Engineer",
            seniority="senior",
            company="Stripe",
            company_style="Deep technical dives, high bar on reliability and API design",
            job_description=JD_BACKEND,
            resume_context=RESUME_BACKEND,
            skills=["Postgres", "Kafka"],
            interview_type="technical",
            round="onsite",
            duration_minutes=20,
        ),
        "strong",
    ),
    Case(
        InterviewParams(
            role="Software Engineer",
            seniority="junior",
            company="Infosys",
            company_style="Fundamentals, OOP, DBMS and a project walkthrough",
            resume_context=RESUME_JUNIOR,
            skills=["Data structures", "SQL"],
            interview_type="technical",
            round="screening",
            duration_minutes=12,
        ),
        "weak",
    ),
    Case(
        InterviewParams(
            role="Senior Data Scientist",
            seniority="senior",
            company="Swiggy",
            company_style="Case-style product analytics plus ML depth",
            job_description=JD_DS,
            resume_context=RESUME_DS,
            skills=["A/B testing", "Forecasting"],
            interview_type="mixed",
            round="onsite",
            duration_minutes=18,
        ),
        "mixed_up_down",
    ),
    Case(
        InterviewParams(
            role="Product Manager",
            seniority="mid",
            company="Atlassian",
            company_style="Values-based behavioural questions, product sense",
            job_description=JD_PM,
            interview_type="behavioral",
            round="final",
            duration_minutes=15,
        ),
        "average",
    ),
    Case(
        InterviewParams(
            role="Staff Software Engineer",
            seniority="principal",
            company="Google",
            company_style="System design at scale, trade-offs, back-of-envelope estimates",
            skills=["Distributed systems"],
            interview_type="system_design",
            round="onsite",
            duration_minutes=25,
        ),
        "strong",
    ),
    Case(
        InterviewParams(
            role="Account Executive",
            seniority="mid",
            company="Razorpay",
            company_style="Role-play discovery and objection handling",
            job_description=JD_SALES,
            interview_type="behavioral",
            round="technical",
            duration_minutes=12,
        ),
        "vague",
    ),
    Case(
        InterviewParams(
            role="Management Consultant",
            seniority="junior",
            company="McKinsey",
            company_style="Interviewer-led case interview with a market-sizing component",
            interview_type="case",
            round="onsite",
            duration_minutes=20,
        ),
        "improving",
    ),
    Case(
        InterviewParams(
            role="Registered Nurse",
            seniority="mid",
            company="Apollo Hospitals",
            company_style="Clinical scenarios and patient-safety judgement",
            job_description=JD_NURSE,
            interview_type="mixed",
            round="screening",
            duration_minutes=12,
        ),
        "average",
    ),
    Case(
        InterviewParams(
            role="Frontend Engineer",
            seniority="mid",
            company="Vercel",
            company_style="Practical React depth and performance",
            job_description=JD_FE,
            skills=["React", "Accessibility"],
            interview_type="technical",
            round="technical",
            duration_minutes=15,
        ),
        "mixed_up_down",
    ),
    Case(
        InterviewParams(
            role="HR Business Partner",
            seniority="senior",
            company="Tata Steel",
            company_style="Situational judgement on employee relations",
            interview_type="hr",
            round="final",
            duration_minutes=12,
        ),
        "strong",
    ),
    Case(
        InterviewParams(
            role="Machine Learning Engineer",
            seniority="mid",
            company="",
            company_style="",
            skills=["Model serving", "PyTorch"],
            interview_type="technical",
            round="technical",
            difficulty="4",
            duration_minutes=15,
        ),
        "weak",
    ),
    Case(
        InterviewParams(
            role="Site Reliability Engineer",
            seniority="senior",
            company="Flipkart",
            company_style="Incident deep dives and capacity planning for sale events",
            skills=["Kubernetes", "Observability"],
            interview_type="mixed",
            round="onsite",
            duration_minutes=18,
        ),
        "improving",
    ),
    Case(
        InterviewParams(
            role="Data Analyst",
            seniority="intern",
            company="Zomato",
            company_style="SQL and business sense, friendly",
            skills=["SQL", "Excel"],
            interview_type="technical",
            round="screening",
            duration_minutes=10,
        ),
        "average",
    ),
    Case(
        InterviewParams(
            role="Engineering Manager",
            seniority="lead",
            company="Microsoft",
            company_style="People leadership, delivery and growth mindset",
            interview_type="behavioral",
            round="final",
            duration_minutes=20,
        ),
        "vague",
    ),
    Case(
        InterviewParams(
            role="Backend Engineer",
            seniority="mid",
            company="Startup (seed stage)",
            company_style="Scrappy, ownership, broad skills",
            job_description=JD_BACKEND,
            interview_type="mixed",
            round="technical",
            language="hi",
            duration_minutes=12,
        ),
        "average",
    ),
    Case(
        InterviewParams(
            role="Security Engineer",
            seniority="senior",
            company="CrowdStrike",
            company_style="Threat modelling and incident response",
            skills=["Threat modelling"],
            interview_type="technical",
            round="onsite",
            duration_minutes=15,
        ),
        "strong",
    ),
    Case(
        InterviewParams(
            role="Business Analyst",
            seniority="junior",
            company="Deloitte",
            company_style="Case plus stakeholder communication",
            interview_type="case",
            round="screening",
            duration_minutes=12,
        ),
        "weak",
    ),
    Case(
        InterviewParams(
            role="Mobile Engineer (Android)",
            seniority="mid",
            company="PhonePe",
            company_style="App performance and offline-first design",
            skills=["Kotlin", "Performance"],
            interview_type="system_design",
            round="technical",
            duration_minutes=15,
        ),
        "mixed_up_down",
    ),
    Case(
        InterviewParams(
            role="Teacher (Mathematics)",
            seniority="mid",
            company="Delhi Public School",
            company_style="Classroom scenarios and pedagogy",
            interview_type="behavioral",
            round="final",
            duration_minutes=10,
        ),
        "improving",
    ),
    Case(
        InterviewParams(
            role="DevOps Engineer",
            seniority="junior",
            company="TCS",
            company_style="Linux, CI/CD basics and a project discussion",
            skills=["Linux", "CI/CD"],
            interview_type="technical",
            round="screening",
            duration_minutes=10,
        ),
        "vague",
    ),
]

PROFILE_TEXT = {
    "strong": "Give strong, specific answers: a concrete example, your own actions, numbers and trade-offs.",
    "average": "Give reasonable but partly generic answers: some specifics, little measurement.",
    "weak": "Give weak answers: short, uncertain, sometimes partly wrong, few specifics.",
    "vague": "Give vague, buzzword-heavy answers that avoid specifics and never mention outcomes.",
}


def profile_for(case: Case, turn: int) -> str:
    p = case.profile
    if p == "mixed_up_down":
        return "strong" if turn % 2 == 0 else "weak"
    if p == "improving":
        return "weak" if turn < 2 else ("average" if turn < 4 else "strong")
    return p


CANDIDATE_SYSTEM = (
    "You are role-playing a job candidate in a spoken mock interview so the interviewer can be tested. "
    "Answer the interviewer's latest question in the first person, out loud, in 40 to 120 words, as plain "
    "speech (no lists, no markdown). Stay consistent with your background. Follow the quality instruction "
    "for this answer exactly; do not mention that you are role-playing."
)


def candidate_answer(llm: ChatLLM, case: Case, st: AgentState, quality: str) -> str:
    p = case.params
    history = "\n".join(f"Interviewer: {t.say}\nYou: {t.answer}" for t in st.turns if t.answer)
    msg = (
        f"Your background: applying for {p.role} ({p.seniority}) at {p.company or 'a company'}. "
        f"Resume: {p.resume_context or 'typical for this role'}\n"
        f"Answer in {'Hindi' if p.language == 'hi' else 'English'}.\n"
        f"Quality instruction: {PROFILE_TEXT[quality]}\n\nConversation so far:\n{history}\n\n"
        f"Interviewer's latest question: {st.turns[-1].say}\nYour answer:"
    )
    for attempt in range(3):
        try:
            text = "".join(
                llm.stream(
                    CANDIDATE_SYSTEM, [{"role": "user", "content": msg}], max_tokens=400, timeout_s=180
                )
            )
            return " ".join(text.split())
        except Exception:  # transient: retry, then give up loudly
            if attempt == 2:
                raise
            time.sleep(2)
    return ""


def run_case(i: int, case: Case, chain: list[ChatLLM], candidate: ChatLLM) -> dict:
    agent = InterviewerAgent(chain, timeout_s=240)
    st = AgentState.new(f"mock-{i:02d}", case.params)
    clock = 1_000_000.0
    turn = agent.next_turn(st, None, now=clock)
    n = 0
    while turn is not None and turn.action != "close" and n < 25:
        quality = profile_for(case, n)
        ans = candidate_answer(candidate, case, st, quality)
        turn.reason = f"{turn.reason} [candidate quality: {quality}]"
        clock += 8 + len(ans.split()) / 2.5  # question + answer at 150 wpm
        turn = agent.next_turn(st, Answer(ans, seconds=len(ans.split()) / 2.5), now=clock)
        n += 1
    return {
        "state": st.to_dict(),
        "attempts": [a.__dict__ for a in agent.attempts],
        "qualities": [profile_for(case, k) for k in range(n)],
    }


def check(result: dict) -> dict:
    st = AgentState.from_dict(result["state"])
    bank = [b for b in bank_texts() if b]
    says = [t.say for t in st.turns]
    overlaps = [s for s in says for b in bank if s.strip().lower() == b.strip().lower() or is_repeat(s, b)]
    follow = [t for t in st.turns if t.action in ("follow_up", "challenge")]
    follow_ok = []
    for t in follow:
        prev = st.turns[t.index - 1].answer or ""
        refs = quote_matches(t.anchor_quote, prev) and bool(
            content_words(t.say) & (content_words(prev) | content_words(t.anchor_quote))
        )
        follow_ok.append(refs)
    diff_ok, transitions = True, []
    for a, b in zip(st.turns, st.turns[1:], strict=False):
        if a.score is None:
            continue
        transitions.append({"score": a.score, "from": a.difficulty, "to": b.difficulty})
        if (a.score >= 4 and b.difficulty < a.difficulty) or (a.score <= 2 and b.difficulty > a.difficulty):
            diff_ok = False
    repeats = [(i, j) for i in range(len(says)) for j in range(i) if is_repeat(says[i], says[j])]
    corrections = sum(len(t.corrections) for t in st.turns)
    return {
        "questions": len(says),
        "bank_overlap": len(overlaps),
        "follow_ups": len(follow),
        "follow_ups_referencing_answer": sum(follow_ok),
        "difficulty_tracks_performance": diff_ok,
        "difficulty_transitions": transitions,
        "difficulty_corrections_by_code": corrections,
        "emergency_questions": sum(t.emergency for t in st.turns),
        "emergency_blueprint": bool(st.blueprint and st.blueprint.emergency),
        "repeats": len(repeats),
        "closed_on_time": st.finished,
        "passed": not overlaps
        and all(follow_ok)
        and diff_ok
        and not repeats
        and not any(t.emergency for t in st.turns)
        and not (st.blueprint and st.blueprint.emergency),
    }


def to_markdown(i: int, case: Case, result: dict, checks: dict) -> str:
    st = AgentState.from_dict(result["state"])
    p = st.params
    bp = st.blueprint
    lines = [
        f"# Mock interview {i:02d}: {p.role} ({p.seniority}) at {p.company or 'unspecified company'}",
        "",
        f"- Type / round: {p.interview_type} / {p.round}; duration {p.duration_minutes} min; language {p.language}; requested difficulty {p.difficulty}",
        f"- Company style: {p.company_style or '-'}",
        f"- Skills to probe: {', '.join(p.skills) or '-'}",
        f"- Candidate profile (simulated): {case.profile}",
        f"- Interviewer LLM: {', '.join(sorted({a['provider'] + ':' + a['model'] for a in result['attempts']}))}",
        f"- Checks: {'PASS' if checks['passed'] else 'FAIL'} "
        + json.dumps({k: v for k, v in checks.items() if k not in ("difficulty_transitions",)}),
        "",
        "## Blueprint",
        bp.summary if bp else "-",
        "",
    ]
    for c in bp.competencies if bp else []:
        lines.append(f"- **{c.name}** ({c.minutes:g} min): {c.why}")
    lines += ["", "## Transcript", ""]
    for t in st.turns:
        tag = f"[{t.action}, {t.competency}, difficulty {t.difficulty}{', EMERGENCY' if t.emergency else ''}]"
        lines.append(f"**Interviewer** {tag}: {t.say}")
        if t.anchor_quote:
            lines.append(f'  - builds on: "{t.anchor_quote}"')
        if t.answer is not None:
            lines.append(f"\n**Candidate**: {t.answer}")
            if t.score is not None:
                lines.append(f"  - interviewer's read: {t.score}/5")
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "docs/evidence/questions"))
    ap.add_argument("--shard", default="0/1", help="i/n: run every n-th case starting at i")
    ap.add_argument("--only", default="", help="e.g. 0-4 or 3")
    args = ap.parse_args()
    chain = chain_from_env()
    if not chain:
        print("No LLM configured (LLM_PROVIDER). Refusing to fabricate transcripts.", file=sys.stderr)
        return 2
    candidate = chain[0]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    si, sn = (int(x) for x in args.shard.split("/"))
    idx = [i for i in range(len(CASES)) if i % sn == si]
    if args.only:
        a, _, b = args.only.partition("-")
        idx = [i for i in idx if int(a) <= i <= int(b or a)]
    summary = []
    for i in idx:
        t0 = time.time()
        result = run_case(i, CASES[i], chain, candidate)
        checks = check(result)
        (out / f"interview-{i:02d}.json").write_text(
            json.dumps({"checks": checks, **result}, indent=1, ensure_ascii=False)
        )
        (out / f"interview-{i:02d}.md").write_text(to_markdown(i, CASES[i], result, checks), encoding="utf-8")
        row = {
            "case": i,
            "seconds": round(time.time() - t0),
            **{k: v for k, v in checks.items() if k != "difficulty_transitions"},
        }
        summary.append(row)
        print("RESULT " + json.dumps(row), flush=True)
    (out / f"summary-shard{si}.json").write_text(json.dumps(summary, indent=1))
    return 0 if all(r["passed"] for r in summary) else 1


if __name__ == "__main__":
    sys.exit(main())
