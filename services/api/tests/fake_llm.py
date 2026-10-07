"""A scripted chat LLM for API tests only (never used for evidence or accuracy claims).

It answers in the formats the agent and evaluator require, so API tests exercise the real
agent code paths (validation, bookkeeping, persistence) without a network call.
"""

from __future__ import annotations

import json
import re

from interview_core.agent.llm import Usage

QUESTIONS = [
    "Walk me through the architecture of the last backend service you owned?",
    "How did you decide on the data model for that service, and what would you change now?",
    "Tell me about a production incident you handled from detection to postmortem?",
    "How do you approach code review when you disagree with a senior colleague?",
    "Describe how you would design rate limiting for a public API serving many tenants?",
    "Explain how you measure whether a performance optimisation actually worked?",
    "Talk me through a time you had to push back on an unrealistic deadline?",
    "How would you migrate a large table with zero downtime?",
]


class FakeInterviewerLLM:
    name, model = "scripted", "scripted-1"

    def __init__(self) -> None:
        self.calls = 0
        self.asked = 0
        self.last_usage = Usage(800, 120)

    def stream(self, system, messages, *, max_tokens=1200, timeout_s=30.0, effort="low"):
        self.calls += 1
        last = messages[-1]["content"]
        everything = "\n".join(m["content"] for m in messages)
        if system.startswith("You are a senior interviewer preparing"):
            m = re.search(r"Duration: (\d+) minutes", everything)
            mins = int(m.group(1)) if m else 20
            sk = re.search(r"Skills to probe: (.*)", everything)
            skills = "" if not sk or sk.group(1).startswith("(none") else sk.group(1)
            each = max(1, (mins - 2) // 3)
            bp = {
                "summary": "Backend depth, incident handling and collaboration.",
                "competencies": [
                    {
                        "id": "c1",
                        "name": "System design",
                        "why": f"JD; probes {skills}" if skills else "JD",
                        "weight": 0.4,
                        "minutes": each,
                        "signals": ["trade-offs"],
                    },
                    {
                        "id": "c2",
                        "name": "Operations",
                        "why": "resume",
                        "weight": 0.3,
                        "minutes": each,
                        "signals": [],
                    },
                    {
                        "id": "c3",
                        "name": "Collaboration",
                        "why": "seniority",
                        "weight": 0.3,
                        "minutes": each,
                        "signals": [],
                    },
                ],
                "opening": "Greet then ask about a recent service.",
                "style_notes": "Direct and friendly.",
                "start_difficulty": 3,
            }
            yield json.dumps(bp)
            return
        if "conforms to this JSON Schema" in system:
            answer = re.search(r"<candidate_input>\n(.*?)\n</candidate_input>", last, re.S)
            words = (answer.group(1) if answer else "").split()
            quote = " ".join(words[:5]) if len(words) >= 5 else ""
            yield json.dumps(
                {
                    "scores": {
                        "relevance": 4,
                        "structure": 3,
                        "depth": 3,
                        "communication": 4,
                        "technical_accuracy": None,
                    },
                    "star": {"situation": True, "task": True, "action": True, "result": bool(words)},
                    "evidence": [{"dimension": "depth", "quote": quote, "comment": "specific"}]
                    if quote
                    else [],
                    "strengths": ["Concrete example."],
                    "improvements": ["Quantify the result."],
                    "follow_up": {"needed": False, "question": ""},
                    "summary": "Solid answer.",
                }
            )
            return
        has_answer = "Candidate's answer" in "".join(m["content"] for m in messages[1:])
        last_answer = (
            {"score": 4, "strengths": "specific", "gaps": "", "vague": False} if has_answer else None
        )
        low = last.lower()
        if "use action open" in low:
            action, say = "open", "Hi, I'm Maya. " + QUESTIONS[0]
        elif "use action wrap_up" in low:
            action, say = "wrap_up", "We're nearly out of time. Do you have any questions for me?"
        elif "use action close" in low:
            action, say = "close", "Thank you for your time today. Goodbye."
        else:
            self.asked += 1
            action, say = "new_topic", QUESTIONS[self.asked % len(QUESTIONS)]
        comp = ["c1", "c2", "c3"][self.asked % 3]
        plan = {
            "last_answer": last_answer,
            "action": action,
            "competency": comp,
            "difficulty": 3,
            "anchor_quote": "",
            "reason": "test",
        }
        yield f"<plan>{json.dumps(plan)}</plan>\n<say>{say}</say>"
