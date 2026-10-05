# Answer evaluation (E6 / E9)

- **Purpose:** explainable feedback on each answer.
- **Method:**
  - LLM rubric scoring (`docs/RUBRIC.md`): relevance, structure (STAR), depth, communication, and technical accuracy for technical questions.
  - Every judgement quotes the answer, and quotes are verified against the transcript. Fabricated quotes trigger a repair or the fallback.
  - Fallback: explainable heuristics with character spans; it never judges technical correctness.
  - Delivery metrics (pace, pauses, fillers) come from ASR word timings.
  - The prototype's nine-factor keyword score is kept as an appendix for continuity, labelled experimental.
- **Label:** all scores show **experimental** until `answer_scoring` shows Spearman ≥ 0.6 and QWK ≥ 0.6 against ≥ 3 trained raters, with no subgroup more than 0.1 below the overall figure; then `SCORING_CALIBRATED=true`.
- **Prohibited outputs:** emotions, personality, confidence as a trait, accent, any protected characteristic.
- **Evaluation:** `answer_scoring` suite (by accent, gender, seniority). **Not measured.**
- **Known risks:** ASR errors on accented speech propagate into content scores (measure WER by accent alongside); verbose answers may be over-rewarded; LLM variance between runs.
