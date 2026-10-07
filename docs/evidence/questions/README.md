# Mock interviews with the real LLM interviewer (P1 evidence)

Each interview was run by `eval/agent_mock_interviews.py` against a real open-weights model. The candidate is simulated by the same model and given a profile (strong, average, weak, vague, mixed or improving). No question comes from a bank, and the checks below are computed by code, not by reading.

Source: [Agent evidence run 37528464751](https://github.com/DeAtHfIrE26/Next-Generation-Virtual-Interview-Training-System/actions/runs/37528464751) on commit 3e0f5b1 (2026-10-06, Qwen2.5-7B-Instruct on Ollama, CPU-only GitHub runners)

## Totals

- Interviews: 20; passed every check: **20/20**
- Questions asked: 279; follow-ups 165, of which 165 quote and ask about the previous answer
- Overlap with the old question bank: 0; repeated questions: 0
- Emergency (template) questions: 0; emergency blueprints: 0
- Questions written by the focused fresh-question call (D19): 8
- Difficulty moved against the candidate's performance: 0 interviews
- LLM calls: 428; accepted-call latency p50 88.7 s, p95 216.7 s (CPU-only CI runner; a GPU or hosted model is far faster)

## Per interview

| # | Role | Type | Qs | Follow-ups (grounded) | Repeats | Emergency | Fresh-call | Result |
|---|---|---|---|---|---|---|---|---|
| [00](interview-00.md) | Backend Engineer (senior) | technical, 20 min, en | 15 | 11 (11) | 0 | 0 | 0 | PASS |
| [01](interview-01.md) | Software Engineer (junior) | technical, 12 min, en | 16 | 10 (10) | 0 | 0 | 0 | PASS |
| [02](interview-02.md) | Senior Data Scientist (senior) | mixed, 18 min, en | 16 | 11 (11) | 0 | 0 | 0 | PASS |
| [03](interview-03.md) | Product Manager (mid) | behavioral, 15 min, en | 9 | 4 (4) | 0 | 0 | 0 | PASS |
| [04](interview-04.md) | Staff Software Engineer (principal) | system_design, 25 min, en | 15 | 11 (11) | 0 | 0 | 0 | PASS |
| [05](interview-05.md) | Account Executive (mid) | behavioral, 12 min, en | 13 | 10 (10) | 0 | 0 | 0 | PASS |
| [06](interview-06.md) | Management Consultant (junior) | case, 20 min, en | 16 | 9 (9) | 0 | 0 | 0 | PASS |
| [07](interview-07.md) | Registered Nurse (mid) | mixed, 12 min, en | 13 | 7 (7) | 0 | 0 | 2 | PASS |
| [08](interview-08.md) | Frontend Engineer (mid) | technical, 15 min, en | 17 | 12 (12) | 0 | 0 | 0 | PASS |
| [09](interview-09.md) | HR Business Partner (senior) | hr, 12 min, en | 9 | 5 (5) | 0 | 0 | 0 | PASS |
| [10](interview-10.md) | Machine Learning Engineer (mid) | technical, 15 min, en | 16 | 8 (8) | 0 | 0 | 0 | PASS |
| [11](interview-11.md) | Site Reliability Engineer (senior) | mixed, 18 min, en | 15 | 10 (10) | 0 | 0 | 0 | PASS |
| [12](interview-12.md) | Data Analyst (intern) | technical, 10 min, en | 12 | 7 (7) | 0 | 0 | 0 | PASS |
| [13](interview-13.md) | Engineering Manager (lead) | behavioral, 20 min, en | 12 | 7 (7) | 0 | 0 | 1 | PASS |
| [14](interview-14.md) | Backend Engineer (mid) | mixed, 12 min, hi | 15 | 3 (3) | 0 | 0 | 3 | PASS |
| [15](interview-15.md) | Security Engineer (senior) | technical, 15 min, en | 10 | 6 (6) | 0 | 0 | 0 | PASS |
| [16](interview-16.md) | Business Analyst (junior) | case, 12 min, en | 19 | 10 (10) | 0 | 0 | 0 | PASS |
| [17](interview-17.md) | Mobile Engineer (Android) (mid) | system_design, 15 min, en | 18 | 11 (11) | 0 | 0 | 1 | PASS |
| [18](interview-18.md) | Teacher (Mathematics) (mid) | behavioral, 10 min, en | 10 | 4 (4) | 0 | 0 | 1 | PASS |
| [19](interview-19.md) | DevOps Engineer (junior) | technical, 10 min, en | 13 | 9 (9) | 0 | 0 | 0 | PASS |

A case fails if any emergency question or blueprint was used, a question repeats an earlier one or the bank, a follow-up does not quote and ask about the previous answer, or difficulty moved against the score.
