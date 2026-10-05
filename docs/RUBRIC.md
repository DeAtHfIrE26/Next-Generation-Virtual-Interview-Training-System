# Answer rubric (v1)

Used by the LLM evaluator (`interview_core.nlp.evaluator.RUBRIC`) and by human raters in the `answer_scoring` evaluation suite. Raters and the model use the same text, so their scores can be compared.

| Dimension | 1 | 3 | 5 |
|---|---|---|---|
| **Relevance** | Off-topic or answers a different question | Partly addresses what was asked | Fully answers the question asked |
| **Structure** | Disorganised; hard to follow the thread | Some order; key parts present but jumbled | Clear flow. For behavioural questions: situation, task, action, result |
| **Depth** | Vague, generic, no specifics | Some specifics | Concrete examples, numbers, trade-offs and reasoning |
| **Communication** | Hard to follow | Understandable with effort | Concise and clear |
| **Technical accuracy** | Incorrect (null for non-technical questions) | Partly correct | Correct and precise |

Scores 2 and 4 sit between the anchors.

## Rules for raters and for the model

- Judge only the words of the answer, as transcribed.
- Do **not** score or comment on personality, emotions, nervousness, confidence as a trait, accent, gender or any protected characteristic.
- Every judgement cites a quote from the answer (the model's quotes are machine-checked against the transcript).
- Overall score = mean of the scored dimensions, rescaled to 0-1.

## Calibration

Scores are shown as **experimental** until the `answer_scoring` suite reports Spearman ≥ 0.6 and quadratic-weighted kappa ≥ 0.6 against the mean of ≥3 trained raters on ≥300 answers, with no subgroup more than 0.1 below the overall figure. After that, set `SCORING_CALIBRATED=true`.
