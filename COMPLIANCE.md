# Compliance status

> **Not legal advice.** This is an engineering record of what the code does and what still needs a lawyer. Every regulatory statement below must be confirmed by qualified counsel for each market before launch. Dates and the status of laws change; check current versions.

## Product positioning

**v1 is a candidate-side coaching tool:** people practise for their own interviews. It is not sold or configured for employers to screen or rank candidates.

- Hiring-decision use is blocked by default (`FEATURE_B2B_HIRING=false`). The terms prohibit it.
- B2B seats exist for colleges, placement cells and bootcamps, for their students' practice only.
- Turning employer screening on changes the regulatory category (see below). It requires the items marked **[B2B hiring]** to be done first.

## What the code does today

| Area | Implementation | Where |
|---|---|---|
| Consent | Per purpose (`data_processing`, `biometric_face`, `biometric_voice`, `store_recordings`, `model_training`), versioned, revocable, enforced server-side for each action | `routers/consent.py` |
| Biometric minimisation | Video is analysed on the device. Server receives landmarks/numbers and short-lived face crops; raw images and recordings are not stored | `apps/web/lib/vision.ts`, `routers/enrollment.py` |
| Biometric storage | Encrypted numeric templates only (AES-256-GCM, KEK in Cloud KMS, bound to owner/kind/model) | `interview_core/crypto` |
| Retention | Templates expire (`BIOMETRIC_RETENTION_DAYS`, default 30); sessions (`SESSION_RETENTION_DAYS`, default 365); raw media 0 days by default; daily purge job | `routers/privacy.py::purge_expired`, Terraform scheduler |
| Withdrawal and erasure | Withdrawing biometric consent deletes the template immediately; account deletion cascades to every personal record; audit log keeps only the event | `routers/consent.py`, `routers/privacy.py` |
| Access and portability | One-click JSON export of everything held | `GET /privacy/export` |
| Purpose limitation | Contact details stripped from resumes before any AI processing; model training only with separate opt-in (not used by any pipeline yet) | `nlp/resume.py` |
| Children | Adults only: sign-up requires confirming age 18+ | `routers/auth.py` |
| Emotion recognition | **Not implemented in any mode.** The prototype's DeepFace emotion code is not used. Prompts forbid inferring emotions, personality or protected traits | `nlp/evaluator.py` rubric, `docs/RUBRIC.md` |
| Explainability | Every score cites the candidate's own words (machine-verified); delivery and gaze feedback cite timestamps; uncalibrated scores are labelled "experimental" | `nlp/evaluator.py`, `report/` |
| Human oversight / no automated decisions | Coaching scores have no consequence. Integrity notices are informational in coaching mode | `security/policy.py` |
| Security | Argon2id passwords, hashed session tokens, CSRF guard, rate limits, CSP, redacted logs, secret scanning in CI | `security.py`, `observability.py`, `tools/secret_scan.py` |
| Third-party processors | LLM, ASR, TTS, payments, hosting, each optional and configured per deployment | `.env.example` |

## India: Digital Personal Data Protection Act 2023 and Rules

| Obligation (summary) | Status |
|---|---|
| Notice before consent, in clear language, listing data and purposes (s.5) | Draft notice: `docs/legal/privacy-policy-draft.md`, consent texts in-app. **Counsel to finalise**, including Eighth Schedule language options if serving users in other Indian languages. |
| Free, specific, informed, unambiguous consent; withdrawal as easy as giving it (s.6) | Implemented per purpose, with toggles in Privacy settings. |
| Consent manager integration (s.6(7)) | Not implemented. **Decide with counsel** whether needed. |
| Reasonable security safeguards; breach notification to the Data Protection Board and each affected person (s.8) | Safeguards listed above. **Breach runbook and notification templates needed** (see below). |
| Erasure when the purpose is served or consent is withdrawn (s.8(7)) | Retention jobs and erasure implemented. **Counsel to confirm retention periods.** |
| Children: verifiable parental consent; no tracking or behavioural monitoring of children (s.9) | Adults-only gate. **Counsel: is self-declaration sufficient, or is age assurance needed?** Placement cells may have under-18 students; their seats must stay adults-only or get a parental-consent flow. |
| Rights: access, correction, erasure, grievance redressal, nomination (ss.11-14) | Access and erasure in-app. Correction is via profile editing (name) and re-enrolment. **Grievance officer contact and nomination process needed.** |
| Significant Data Fiduciary duties (DPO, DPIA, audits), if notified | **Counsel to assess.** Biometric processing at scale raises the likelihood. |
| Cross-border transfer | Hosting in asia-south1 (Mumbai). LLM and ASR vendors may process data abroad. **Counsel to check against any restricted-country list and record vendor regions.** |
| Commencement | The DPDP Rules were notified in November 2025 with staged commencement. **Confirm current effective dates.** Until full commencement, the IT Act SPDI Rules 2011 (which treat biometric data as sensitive personal data) also apply. |

## European Union (if offered to EU users)

- **GDPR:** biometric data used for identification is special-category data (Art. 9). Explicit consent is implemented. A DPIA is likely required. Plan for SCCs for processors outside the EU. **Lawyer and EU representative needed.**
- **EU AI Act:**
  - Emotion recognition in the workplace or in education institutions is **prohibited** (Art. 5(1)(f)). This product does not do emotion recognition in any mode, and must not, including for university and placement-cell customers.
  - AI used for recruitment or selection is **high-risk** (Annex III). **[B2B hiring]** requires: a risk management system, data governance, technical documentation, logging, human oversight, accuracy/robustness/cybersecurity requirements, conformity assessment and registration. Check current application dates; amendments to high-risk timelines have been proposed.
  - Remote biometric identification rules: the 1:1 verification used here (is this the enrolled user?) is generally treated differently from identification. **Counsel to confirm classification.**
  - Transparency: users are told they are interacting with an AI interviewer (landing page, room). **Confirm the wording meets Art. 50.**

## United States (if offered)

- **NYC Local Law 144 [B2B hiring]:** an automated employment decision tool used for NYC candidates needs an annual independent bias audit, a published summary, and candidate notice at least 10 business days in advance. The eval harness already reports per-subgroup metrics, but an LL144 audit needs real hiring-context data and an independent auditor.
- **Biometric privacy laws** (Illinois BIPA, Texas CUBI, Washington): a written release before collection, a public retention and destruction schedule, and no sale of biometrics. The consent text and retention config cover the mechanics. **Counsel to adapt the release language and publish the schedule** before enabling biometrics for US users. Consider disabling biometric enrolment by region.

## Gaps that need people, not code

1. **Patent and IP ownership.** The patent applicant is VIT. Assignment or licence, inventor agreements, and ownership of student-written code under VIT policy. (Largest blocker.)
2. Final privacy notice, terms, consent texts, cookie notice (only a first-party session cookie is used).
3. Grievance officer, DPO (if required), breach response runbook with 72-hour-style internal timelines and Board/user notification templates.
4. Vendor DPAs and data-region records for each enabled processor (LLM, ASR, TTS, payments, hosting, error tracking).
5. Age-assurance decision (s.9).
6. Commercial model licences: face embedding (InsightFace commercial licence or another licensed model), any neural avatar model; check SpeechBrain and other model cards.
7. **[B2B hiring]** AI Act conformity, LL144 audit, adverse-impact analysis on real data, human-review workflow. Keep `FEATURE_B2B_HIRING=false` until done.
8. Accessibility audit (WCAG 2.2 AA) and an Indian-languages roadmap.

## Breach runbook (engineering steps; legal timelines to be set by counsel)

1. Contain: rotate the affected credentials (Secret Manager), revoke sessions (`DELETE FROM auth_sessions`), disable the affected integration.
2. Assess: audit log, Cloud Logging, KMS access logs (template unwraps are individually logged).
3. Notify: the Data Protection Board and affected users per DPDP Rules timelines; other regulators as applicable.
4. Record: incident report in the audit store; post-incident review.
