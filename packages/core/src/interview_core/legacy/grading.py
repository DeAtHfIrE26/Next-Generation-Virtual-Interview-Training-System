"""Prototype 9-factor interview grading (E6 evaluation / E9 transcript analysis).

Source: ``legacy/desktop/main.py`` ``grade_interview_with_breakdown`` (L502-870).
The prototype read engagement, warnings and role from module globals; here they are
explicit arguments. Output is identical for identical inputs.

Known defects preserved on purpose (fixed in :mod:`interview_core.nlp.heuristics`):
- filler and keyword counts use substring matching ("um" matches "summary").
- the code-quality factor returns a neutral 0.5 for non-technical roles.
"""

from __future__ import annotations

import re
from typing import Any

TECH_ROLE_MARKERS = (
    "developer", "engineer", "programmer", "software", "data",
    "analyst", "scientist", "architect", "devops", "sde",
)

TECH_WEIGHTS = {"depth": 12.0, "clarity": 12.0, "domain": 18.0, "confidence": 12.0,
                "probsolve": 18.0, "teamwork": 10.0, "codequal": 18.0}
NONTECH_WEIGHTS = {"depth": 15.0, "clarity": 15.0, "domain": 18.0, "confidence": 15.0,
                   "probsolve": 15.0, "teamwork": 15.0, "codequal": 7.0}

EXPLANATION = (
    "Multi-factor Scoring:\n"
    "1) Depth of Response: Evaluates thoroughness and detail level\n"
    "2) Clarity & Organization: Measures articulation and structured communication\n"
    "3) Domain Relevance: Assesses relevance to job role and industry\n"
    "4) Confidence: Evaluates confidence level and conviction\n"
    "5) Problem-Solving: Measures analytical approach and solution-orientation\n"
    "6) Teamwork: Assesses collaboration indicators and team-oriented mindset\n"
    "7) Technical Quality: Evaluates technical expertise and precision\n"
    "8) Engagement: Factors in visual attentiveness during interview\n"
    "9) Behavioral Warnings: Accounts for professional conduct issues\n"
    "These factors are weighted based on job role, combined into an average score, "
    "and adjusted for engagement and warnings."
)


def is_technical_role(job_role: str | None) -> bool:
    role_lower = job_role.lower() if job_role else ""
    return any(tech in role_lower for tech in TECH_ROLE_MARKERS)


def measure_depth(response: str) -> float:
    wc = len(response.split())
    if wc < 15:
        length_score = 0.2
    elif wc < 30:
        length_score = 0.4
    elif wc < 50:
        length_score = 0.6
    elif wc < 80:
        length_score = 0.8
    else:
        length_score = 1.0
    detail_indicators = ["for example", "specifically", "in detail", "to elaborate",
                         "for instance", "such as", "in particular"]
    detail_count = sum(response.lower().count(ind) for ind in detail_indicators)
    detail_score = min(1.0, detail_count / 3.0)
    return (length_score * 0.7) + (detail_score * 0.3)


def measure_clarity(response: str) -> float:
    resp_lower = response.lower()
    wc = len(response.split()) + 1e-6
    filler_words = ["um", "uh", "er", "ah", "uhm", "like ", "basically", "i mean", "you know"]
    filler_ratio = sum(resp_lower.count(fw) for fw in filler_words) / wc
    if filler_ratio < 0.01:
        clarity_factor = 1.0
    elif filler_ratio < 0.03:
        clarity_factor = 0.9
    elif filler_ratio < 0.05:
        clarity_factor = 0.7
    else:
        clarity_factor = 0.5
    basic_connectors = ["first", "second", "third", "next", "finally", "therefore", "thus", "because",
                        "however", "although", "consequently", "furthermore", "moreover", "in addition"]
    advanced_connectors = ["to summarize", "in conclusion", "as a result", "on the other hand",
                           "for this reason", "to illustrate", "in contrast", "similarly"]
    basic_conn_count = sum(resp_lower.count(c) for c in basic_connectors)
    adv_conn_count = sum(resp_lower.count(c) for c in advanced_connectors)
    connector_score = min(1.0, (basic_conn_count + adv_conn_count * 2) / 5.0)
    return (clarity_factor * 0.6) + (connector_score * 0.4)


def measure_domain_relevance(response: str, ctx: str, technical: bool) -> float:
    ctx_tokens = set(re.findall(r"[a-zA-Z]{4,}", ctx.lower()))
    resp_tokens = set(re.findall(r"[a-zA-Z]{4,}", response.lower()))
    base_score = min(1.0, len(ctx_tokens & resp_tokens) / 7.0)
    if technical:
        tech_terms = ["algorithm", "database", "framework", "architecture", "development",
                      "testing", "deployment", "optimization", "system", "software",
                      "api", "cloud", "git", "code", "programming", "function"]
        tech_count = sum(response.lower().count(t) for t in tech_terms)
        tech_score = min(1.0, tech_count / 5.0)
        return (base_score * 0.6) + (tech_score * 0.4)
    return base_score


def measure_confidence(response: str) -> float:
    resp_lower = response.lower()
    wc = len(response.split()) + 1e-6
    disclaimers = ["maybe", "not sure", "i guess", "i think", "probably", "might",
                   "perhaps", "sort of", "kind of", "somewhat", "possibly", "i'm not certain"]
    disc_ratio = sum(resp_lower.count(d) for d in disclaimers) / wc
    if disc_ratio < 0.01:
        uncertainty_score = 1.0
    elif disc_ratio < 0.03:
        uncertainty_score = 0.8
    elif disc_ratio < 0.05:
        uncertainty_score = 0.6
    else:
        uncertainty_score = 0.4
    confidence_phrases = ["i am confident", "i am certain", "definitely", "absolutely",
                          "without doubt", "i am sure", "i know", "i strongly believe"]
    confidence_boost = min(0.3, sum(resp_lower.count(cp) for cp in confidence_phrases) * 0.1)
    return min(1.0, uncertainty_score + confidence_boost)


STRUCTURED_PATTERNS = [
    "first.+then.+finally", "step.+next.+finally",
    "identify.+analyze.+solve", "understand.+approach.+implement",
    "define.+design.+develop", "problem.+solution.+result",
]


def measure_problem_solving(response: str) -> float:
    keywords = ["approach", "solution", "method", "algorithm", "strategy", "plan",
                "steps", "test", "analyze", "evaluate", "implement", "design", "debug"]
    resp_lower = response.lower()
    basic_score = min(1.0, sum(1 for kw in keywords if kw in resp_lower) / 5.0)
    pattern_bonus = 0.3 if any(re.search(p, resp_lower) for p in STRUCTURED_PATTERNS) else 0
    return min(1.0, basic_score + pattern_bonus)


def measure_teamwork(response: str) -> float:
    teamwork_terms = ["team", "collaborate", "we ", "our ", "together", "partner",
                      "collective", "cooperation", "colleagues", "group", "joint"]
    resp_lower = response.lower()
    found = sum(resp_lower.count(t) for t in teamwork_terms)
    basic_score = 0.2 if found == 0 else min(1.0, found / 4.0)
    stories = ["worked with team", "collaborated on", "our team achieved",
               "we implemented", "team project", "cross-functional"]
    story_bonus = 0.2 if any(s in resp_lower for s in stories) else 0
    return min(1.0, basic_score + story_bonus)


def measure_code_quality(response: str, technical: bool) -> float:
    if not technical:
        return 0.5
    basic_terms = ["function", "class", "variable", "data structure", "code",
                   "algorithm", "performance", "complexity", "optimize"]
    advanced_terms = ["big-o", "time complexity", "space complexity", "edge case",
                      "exception handling", "testing strategy", "refactoring",
                      "design pattern", "asynchronous", "concurrent"]
    resp_lower = response.lower()
    basic_count = sum(1 for t in basic_terms if t in resp_lower)
    adv_count = sum(1 for t in advanced_terms if t in resp_lower)
    return min(1.0, (basic_count + adv_count * 2) / 8.0)


def engagement_factor(eye_away_count: int | None, total_eye_checks: int | None) -> float:
    if eye_away_count is None or total_eye_checks is None or total_eye_checks == 0:
        return 1.0
    away_ratio = eye_away_count / float(total_eye_checks)
    return max(0.2, 1.0 - away_ratio)


def warning_penalty(warning_count: int) -> int:
    if warning_count <= 2:
        penalty = warning_count * 2
    elif warning_count <= 5:
        penalty = 4 + ((warning_count - 2) * 3)
    else:
        penalty = 13 + ((warning_count - 5) * 4)
    return min(30, penalty)


def grade_interview_with_breakdown(
    transcript: str,
    context: str = "",
    *,
    job_role: str | None = None,
    eye_away_count: int | None = None,
    total_eye_checks: int | None = None,
    warning_count: int = 0,
) -> tuple[int, dict[str, Any], str]:
    """Return ``(final_score, breakdown, summary_message)`` exactly as the prototype did."""
    technical = is_technical_role(job_role)
    candidate_lines = [ln for ln in transcript.split("\n") if ln.strip().startswith("Candidate:")]
    if not candidate_lines:
        return 0, {"explanation": "No candidate responses found."}, "No candidate lines found."

    eng = engagement_factor(eye_away_count, total_eye_checks)
    weights = TECH_WEIGHTS if technical else NONTECH_WEIGHTS
    total_score = 0.0
    per_response = []
    for line in candidate_lines:
        response = line.replace("Candidate:", "").strip()
        f = {
            "depth": measure_depth(response),
            "clarity": measure_clarity(response),
            "domain": measure_domain_relevance(response, context, technical),
            "confidence": measure_confidence(response),
            "probsolve": measure_problem_solving(response),
            "teamwork": measure_teamwork(response),
            "codequal": measure_code_quality(response, technical),
        }
        # Same summation order as the prototype, so floating-point results match bit for bit.
        single = (f["depth"] * weights["depth"] + f["clarity"] * weights["clarity"]
                  + f["domain"] * weights["domain"] + f["confidence"] * weights["confidence"]
                  + f["probsolve"] * weights["probsolve"] + f["teamwork"] * weights["teamwork"]
                  + f["codequal"] * weights["codequal"])
        total_score += single
        per_response.append({
            "response_snippet": response[:60] + ("..." if len(response) > 60 else ""),
            "depth": round(f["depth"], 2),
            "clarity": round(f["clarity"], 2),
            "domain_relevance": round(f["domain"], 2),
            "confidence": round(f["confidence"], 2),
            "problem_solving": round(f["probsolve"], 2),
            "teamwork": round(f["teamwork"], 2),
            "code_quality": round(f["codequal"], 2),
            "response_score": round(single, 2),
        })

    n = len(candidate_lines)
    avg = total_score / n if n else 0.0
    adjusted = (avg * 0.85) + (avg * 0.15 * eng)
    base_score = min(100, max(0, adjusted))
    penalty = warning_penalty(warning_count)
    final_score = max(0, base_score - penalty)
    breakdown = {
        "average_score_before_engagement": round(avg, 2),
        "engagement_factor": round(eng, 2),
        "base_score": round(base_score),
        "warning_penalty": penalty,
        "final_score": round(final_score),
        "per_response_details": per_response,
        "explanation": EXPLANATION,
        "is_technical_role": technical,
    }
    summary = (
        f"Final Score: {round(final_score)}/100. "
        f"Base Score (avg + engagement) = {round(base_score)}, "
        f"Warning Penalty = {penalty}. "
        f"Score based on {n} responses with role-specific weighting."
    )
    return round(final_score), breakdown, summary
