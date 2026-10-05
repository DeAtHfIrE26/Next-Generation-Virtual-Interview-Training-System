# Interviewer question generation (E6)

- **Purpose:** personalised, adaptive practice questions.
- **Method:**
  - The configured LLM (`LLM_PROVIDER`; default Anthropic `claude-opus-5-5` with structured JSON-schema output and server-side refusal fallback) receives role, seniority, the redacted resume, the JD, previous questions and the last answer summary.
  - Output must validate against `question.v1`. It must match the planned category, stay within ±1 difficulty, and not repeat earlier questions (Jaccard < 0.6). Otherwise one repair attempt, then the deterministic 59-question bank.
  - Difficulty moves ±1 per answer (≥ 0.75 up, < 0.40 down).
  - Untrusted text is delimited and declared as data.
- **Safety rules in the prompt:** no questions about age, marital status, religion, caste, health, disability, pregnancy, nationality or other protected characteristics.
- **Evaluation:** `question_relevance` (expert ratings) and `llm_schema_validity` (from production logs via `/admin/export`). **Not measured.**
- **Known risks:** question quality varies by provider; domain coverage of the fallback bank is limited (software, data, product, design, business, general).
