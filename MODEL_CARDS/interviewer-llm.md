# Interviewer agent (E6)

- **Purpose:** a live interviewer that plans and asks every question for this candidate. There is no question bank.
- **Method** (`interview_core.agent`):
  - **Blueprint:** competencies, the reason for each, time allocation and starting difficulty, built from the full parameter set (role, seniority, company and style, JD, redacted resume, skills, type, round, difficulty, language, duration, persona).
  - **Per turn:** the model writes an assessment of the last answer, an action (open, follow-up, challenge, new topic, revisit, wrap-up or close), a competency, a difficulty, an anchor quote and what to say.
  - **Enforced by code, whatever the model does:**
    - difficulty moves at most one step and follows the score;
    - follow-ups and challenges must quote the candidate's last answer verbatim;
    - no repeated or rephrased questions;
    - the interview wraps up and closes on time;
    - spoken output is plain text.
- **Providers:**
  - Anthropic (default model `claude-opus-5-5`, prompt caching, server-side refusal fallback).
  - Gemini.
  - Any OpenAI-compatible endpoint, for example Ollama with Qwen2.5-7B-Instruct (Apache-2.0).

  Gemini and OpenAI-compatible endpoints get grammar-constrained JSON (D8). Repairs narrow the grammar to fix the specific failure (D11).
- **Fallback:** retry → two repairs → alternate provider → a flagged backup question (WARNING log, metering record, UI badge, report note). Tests and the evidence harness fail when it fires.
- **Safety rules in the prompt:** no questions about protected characteristics; no comments on accent, appearance or emotions; untrusted text (resume, JD, answers) is delimited and declared as data.
- **Evidence:**
  - `eval/agent_mock_interviews.py` runs 20 scripted cases with a simulated candidate against a real LLM in CI. Transcripts go to `docs/evidence/questions/`.
  - It checks: no overlap with the old bank, follow-ups that quote the answer, difficulty tracking, no repeats, closing on time, and no backup questions.
  - Measured on Qwen2.5-7B (CPU): 26–58 s per call. Pass rates are recorded per run in that folder.
- **Not measured:** question relevance as rated by experienced interviewers (`question_relevance` suite).
- **Known risks:** small local models follow the protocol less reliably than hosted frontier models, and every failure is visible as a backup question.
