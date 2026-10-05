# Claim Map (PROVISIONAL)

> **Status: provisional.** The filed claims have not been supplied. This map uses the nine system elements in the published **abstract** of application 202541122226, plus field-of-invention para. [0003]. See `docs/patent/SOURCES.md`. When the claims arrive, every row gets re-keyed to its claim number, and any dependent-claim detail missing here is added.

## Current implementation (after milestones M0–M8)

Every element is implemented in the product code and covered by tests. The prototype algorithm for each element is preserved behaviour-for-behaviour in `packages/core/src/interview_core/legacy/` and pinned by characterization tests that run the original prototype code side by side (`packages/core/tests/characterization/`). **Accuracy has not been measured for any element yet, because no consented evaluation data exists** (see `docs/EVAL_REPORT.md`). "IMPLEMENTED" here means the mechanism works on real inputs and is tested; it is not an accuracy claim.

| # | Element | Product implementation | Prototype preserved in | Status | Tests |
|---|---|---|---|---|---|
| E1 | UI receiving candidate information and job roles | `apps/web/app/interview/new`, `services/api/.../routers/sessions.py::create_session` (role, seniority, JD, resume PDF) | `legacy/desktop/main.py` (Tkinter) | IMPLEMENTED | API flow tests, Playwright E2E |
| E2 | Facial recognition: capture facial samples, verify identity | `interview_core.face` (multi-sample embedding template, outlier rejection, calibrated threshold, quality gates, active liveness), `routers/enrollment.py::enrol_face`, periodic `routers/sessions.py::face_check`; on-device capture in `apps/web/lib/vision.ts` | `legacy/face_hist.py` | IMPLEMENTED (needs a commercially licensed embedding model configured to run; returns 503 otherwise) | `test_biometrics.py`, `test_liveness.py`, `test_enrollment.py` |
| E3 | Voice authentication: record references, real-time matching | `interview_core.voice` (prompted-phrase enrolment, per-utterance matching, enrolment-bypass fixed), `routers/enrollment.py::enrol_voice`, `routers/sessions.py::submit_answer` | `legacy/voice.py` | IMPLEMENTED (needs a speaker model configured) | `test_biometrics.py`, `test_enrollment.py` |
| E4 | Lip-sync verification: analyse mouth movements, detect speech-authenticity mismatches | `interview_core.lipsync.avsync` (temporal audio-visual correlation, offset, prominence, still-mouth and silent-motion flags), browser mouth series `apps/web/lib/visionMath.ts`, `routers/sessions.py::submit_answer` | `legacy/lipsync.py` (could never fire) | IMPLEMENTED | `test_avsync.py` (incl. "fires where prototype cannot"), API flow tests |
| E5 | Eye tracking: gaze direction and engagement | `interview_core.gaze` (iris H+V, head yaw/pitch, calibration, smoothing, per-answer observable stats), browser `visionMath.ts` (parity-tested) | `legacy/gaze.py` | IMPLEMENTED | `test_gaze_policy.py`, `visionMath.test.ts` |
| E6 | NLP: parse resumes, generate personalised questions, evaluate responses with transformer models | `interview_core.nlp` (`resume`, `interviewer`, `evaluator`, `structured`, `providers`), schemas in `nlp/schemas/` | `legacy/resume.py`, `legacy/grading.py` | IMPLEMENTED (transformer evaluation when an LLM is configured; heuristic fallback otherwise, labelled) | `test_nlp.py` |
| E7 | Security monitoring: unauthorised devices, preventing assistance | browser phone detection (MediaPipe EfficientDet-Lite, Apache-2.0) and face count; `interview_core.security.policy` (per-type debounce, coaching vs proctored); `routers/sessions.py::post_events` | `legacy/security.py` | IMPLEMENTED | `test_gaze_policy.py`, API flow tests |
| E8 | Technical assessment: present coding challenges, give feedback | `interview_core.codeexec` (hidden tests, Judge0 sandbox, step-limited SQLite), `routers/sessions.py::run_code`, room code panel | `legacy/desktop/main.py` Judge0 calls | IMPLEMENTED (non-SQL languages need a Judge0 instance) | `test_speech_code.py`, API flow tests |
| E9 | Performance evaluation: analyse transcripts, generate comprehensive reports | `interview_core.report.build_report`, `interview_core.delivery`, `routers/sessions.py::finish`, `apps/web/components/ReportView.tsx`; the prototype nine-factor score is still computed in every report's appendix | `legacy/grading.py` | IMPLEMENTED (no mock data) | API flow tests, E2E |
| F1 | Real-time behavioural monitoring | browser vision loop at 12 fps, events batched every 2 s | `legacy/desktop/main.py::monitor_webcam` | IMPLEMENTED | E2E |

Changes of method, each staying within the abstract's wording, are listed in `docs/PLAN.md` §d. **Before merging, re-key this table to the filed claims** (they have not been supplied).

## Baseline: the prototype as found (before M0)

**Repo key**
- **NG** = `DeAtHfIrE26/Next-Generation-Virtual-Interview-Training-System` @ `d821924`, file `main.py`
- **FT** = `DeAtHfIrE26/futuristic-ai-interviewer` @ `358ea09`. Files are named per row.

**Status key**
- **IMPLEMENTED:** does what the element says, on real inputs.
- **PARTIAL:** real code, but it only covers part of the element or is too weak to do the job.
- **STUBBED/MOCKED:** returns fixed or fake results, or is never called.
- **MISSING:** no code.

**Fakes** are flagged with ⚠ (fixed scores, random data, `sleep` + "Passed", and so on).

| # | Element (plain language) | Where (best existing implementation first) | Status | Notes and fakes |
|---|---|---|---|---|
| E1 | UI that takes candidate info (resume) and job role | NG `main_app` L3417–3632, `browse_resume` L3282, `start_interview` L3296–3397. FT `GPTUpdate.py` `main_app` L3784+ | IMPLEMENTED (desktop Tkinter) | Single-user desktop app only. No accounts or persistence. |
| E2a | Capture facial samples | FT `GPTUpdate.py` `capture_face_samples` L1271–1446 (InsightFace). NG `capture_face_samples` L1062–1162 | IMPLEMENTED | FT rejects frames containing more than one face. Samples are held in memory. |
| E2b | Verify candidate identity from face | FT `GPTUpdate.py` `initialize_insightface` L1229–1256, `check_same_person_and_phone` L1524–1690 (ArcFace `buffalo_l` embeddings). NG `compare_face_hist`/`predict_face` L1188–1220 | FT: IMPLEMENTED. NG: PARTIAL | NG's "LBPH" is really a grayscale **intensity-histogram correlation**, which cannot tell identities apart. FT uses `buffalo_l` weights, which are **non-commercial**. No liveness anywhere. ⚠ FT `interview_bot.py` L144 loads `anti_spoof.pth`, a file that doesn't exist. |
| E3a | Record a voice reference | NG `record_voice_reference` L2504–2719. FT `GPTUpdate.py` `record_voice_reference` L2538+ | IMPLEMENTED | ⚠ NG `record_audio` L2421–2422: if there's no reference, the **first interview answer silently becomes the reference**, which bypasses enrolment. |
| E3b | Real-time voice matching | NG `compute_voice_embedding` L919–938 (pyannote, MFCC fallback) used in `record_audio` L2423–2430. FT `detect_voice_spoofing` L2449–2519 (Resemblyzer) | IMPLEMENTED | Fixed threshold of 0.8 with no calibration. NG's MFCC-mean fallback is not a speaker embedding. FT's "synthetic speech" check is a spectral-flatness heuristic. ⚠ FT `interview_bot.py` L186–188 sets `voice_antispoofing = True` with no model. |
| E4 | Lip-sync verification: analyse mouth movement and detect when it doesn't match the speech | NG `verify_lip_sync` L948–974, `compute_mouth_opening` L1002–1018, used in `record_audio` L2433–2465. FT `GPTUpdate.py` `record_voice_reference` L2556–2600. FT `interview_bot.py` `calculate_lip_sync_score` L2081–2126, `verify_lip_sync` L1354–1374 | **STUBBED/MOCKED (in effect)** | ⚠ NG `verify_lip_sync` uses **one frame**, ignores the audio, and returns `max(0.8, …)`. The caller warns only below 0.8, so it **can never fire**. NG's post-answer check (L2459–2465) measures one frame *after* speech ends. FT registration counts the fraction of frames with an open mouth (PARTIAL). ⚠ FT `interview_bot.verify_lip_sync` is `sleep(1.5)` then "Passed". FT `calculate_lip_sync_score` is a single-frame open-mouth vs. volume rule. ⚠ `load_lip_sync_model()` is a stub in both repos. **Nothing correlates the audio signal with mouth motion over time**, which is what "speech authenticity mismatch" requires. |
| E5 | Eye tracking: gaze direction and engagement | NG `detect_eye_gaze` L976–1000 (MediaPipe iris, horizontal ratio, fixed ±0.20), counters in `monitor_webcam`. FT `face_processor.py` `_calculate_attention` L390–444 (adds EAR) | PARTIAL | Horizontal gaze only. No head pose, no per-user calibration, and never validated. ⚠ The report's eye "series" is random noise around one ratio (NG L1670–1676). |
| E6a | Parse resumes | NG `parse_resume` L872–880 (PyMuPDF), `extract_candidate_name` L882, `build_context` L890–904 | IMPLEMENTED | Raw text only, with no section structure. FT `language_model.py` L127–233 adds a section/skills extractor. PyMuPDF is **AGPL**. |
| E6b | Generate personalised interview questions | NG `generate_interview_question` L345–470 (Mistral `mistral-large-latest`), `generate_unique_question_list` L1445–1515 | IMPLEMENTED | No output schema or validation. Falls back to `random.choice` templates (L460). ⚠ FT `GPTUpdate.py` L418–616 is mostly random templates. |
| E6c | Adaptive follow-ups based on previous answers | NG `generate_followup_question` L481–500, used in `interview_loop` L2741+ | PARTIAL | Follow-ups exist. There is no difficulty adaptation and no answer-quality signal driving it. |
| E6d | Evaluate responses **using transformer-based models** | NG `evaluate_response` L332–343 (LLM free-text feedback). Numeric scoring: `grade_interview_with_breakdown` L502–870 | PARTIAL | The numeric score is **keyword and length heuristics**, not a transformer (for example, `"um"` substring counting also matches "summary"). The LLM only writes prose feedback. Never calibrated. |
| E7a | Detect unauthorised devices | NG `detect_phone_in_frame` L1251–1268 (YOLOv8n, COCO "cell phone"), `monitor_webcam` L1350–1387 | IMPLEMENTED | `yolov8n.pt` is **AGPL-3.0**. Precision and recall never measured. |
| E7b | Prevent assistance (other person or voice present) | NG `check_same_person_and_phone` L1286–1348 (more than one face fails the check), `monitor_background_audio` L1020–1054 | IMPLEMENTED (heuristic) | Background audio is an RMS threshold, not speaker detection. One shared `warning_count` mixes every warning type (3 voice "mismatches", in any mix with other warnings, end the session). |
| E8 | Coding challenges with feedback | NG Judge0 `create_submission_judge0`/`get_submission_result_judge0` L3109–3154, `run_code` L3175–3220, local SQL L3079–3107, `submit_challenge` L3222–3243 | IMPLEMENTED | Hard-coded RapidAPI key (leaked). The SQL runs in-process. Feedback is LLM prose only, with no test cases. |
| E9a | Analyse transcripts | NG `grade_interview_with_breakdown` L502–870, `summarize_all_responses` L472 | PARTIAL | See E6d. |
| E9b | Generate comprehensive reports | NG `generate_pdf_report` L1866–2220, `create_similarity_matrix` L1611–1737, `generate_recommendations` L2252–2400 | PARTIAL, ⚠ MOCKED parts | ⚠ L1647–1676: when real data is missing, voice, face and eye similarity series are **generated with `random.uniform`**. ⚠ `generate_reports.py` creates the ~80 committed report PDFs and resumes with **random scores and names**. They are not real sessions. |
| F1 | "Real-time behavioural monitoring" (field, [0003]) | NG `monitor_webcam` L1350, `update_camera_view` L1389 | IMPLEMENTED (desktop) | |

## Not patent elements, but asked for or advertised

| Item | Where | Status | Note |
|---|---|---|---|
| **Generative lip-synced avatar interviewer** | NG `animate_bot` L1423–1443, `create_gif.py` | **MISSING** | The "avatar" is a 16 KB GIF of a circle that loops during offline TTS. `lip_sync_frames` is always `None`. **Not in the patent abstract.** In both the patent and the paper, "lip-sync" means *verifying the candidate* (E4). Wav2Lip appears only in a comment (FT `chatGPT.py` L16). |
| Emotion recognition | FT `face_processor.py` L291–302 (DeepFace) | IMPLEMENTED (FT only; that file imports a `config` module that doesn't exist) | **Not in the patent abstract.** The paper reports "93%". Plan: off by default and never shown in any employer mode (see PLAN §f). |
| ASR | `recognize_google` (both repos) | IMPLEMENTED | Unofficial free Google endpoint, not licensed for commercial use. No word timestamps. |
| TTS | `pyttsx3` (both) | IMPLEMENTED | OS voices. No visemes or timestamps. |
