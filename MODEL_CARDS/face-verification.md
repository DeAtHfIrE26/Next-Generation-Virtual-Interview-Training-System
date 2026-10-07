# Face verification (E2)

- **Purpose:** check that the person in a practice session is the person who enrolled. 1:1 verification, never 1:N identification.
- **Method:** multi-sample enrolment (5+ crops that pass quality gates), unit-norm mean embedding after outlier rejection, cosine similarity against a calibrated threshold. Periodic checks during the session. Code: `interview_core.face`.
- **Model:** pluggable `FaceEmbedder`. The ONNX adapter accepts any ArcFace-style 112×112 model. **No model ships.** The prototype's InsightFace `buffalo_l` weights are non-commercial and are excluded.
- **Inputs/outputs:** RGB face crop → embedding; template plus probe → score in [-1, 1] and match/mismatch, or "uncalibrated" when no threshold is set.
- **Intended use:** practice-session integrity, with the user's explicit consent. **Out of scope:** identifying unknown people, any decision about employment.
- **Threshold:** `FACE_MATCH_THRESHOLD`, chosen on the calibration split for the target FAR (plan: ≤ 0.1% at FRR ≤ 3%). Without it, no decision is made.
- **Evaluation:** `face_verification` suite (FAR/FRR/EER by gender, age band, Monk skin tone, glasses, device, lighting). **Not measured.**
- **Known risks:** demographic differentials common to face recognition; lighting and low-end webcams; glasses. Subgroup gaps must be measured before enabling enforcement in proctored mode.
- **Data handling:** crops are decoded in memory and discarded. Templates are encrypted (KMS) and expire.
