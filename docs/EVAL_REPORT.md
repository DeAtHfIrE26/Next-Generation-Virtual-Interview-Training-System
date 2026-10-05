# Evaluation Report

Generated 2026-10-05 16:37 UTC at commit `3e8dcfb` by `python -m eval_harness run`.

**Rule:** a number appears here only if it was measured on a real, consented evaluation set described by a manifest. Synthetic smoke runs check that pipelines execute and never report accuracy. Systems named `legacy_*` are the prototype algorithms (the "before" numbers); the others are the upgraded implementations ("after").

## Summary

| Suite | Status | Systems |
|---|---|---|
| face_verification | NOT MEASURED: no evaluation data yet | - |
| liveness | NOT MEASURED: no evaluation data yet | - |
| speaker_verification | NOT MEASURED: no evaluation data yet | - |
| lipsync | NOT MEASURED: no evaluation data yet | - |
| gaze | NOT MEASURED: no evaluation data yet | - |
| asr | NOT MEASURED: no evaluation data yet | - |
| question_relevance | NOT MEASURED: no evaluation data yet | - |
| answer_scoring | NOT MEASURED: no evaluation data yet | - |
| llm_schema_validity | NOT MEASURED: no evaluation data yet | - |
| device_detection | NOT MEASURED: no evaluation data yet | - |
| latency | NOT MEASURED: no evaluation data yet | - |
| avatar_sync | NOT MEASURED: no evaluation data yet | - |

## face_verification

E2 face verification: FAR / FRR / EER, by subgroup.

**Status:** NOT MEASURED: no evaluation data yet

**Data needed:** >=100 subjects; per subject 1 enrolment session (>=15 aligned face crops) and >=10 probe crops across 2 lighting conditions and >=2 devices; optional self-reported subgroups.
- No manifest at eval/data/face_verification/manifest.jsonl. Not measured.

## liveness

E2 liveness: APCER per attack type and BPCER (ISO/IEC 30107-3 style).

**Status:** NOT MEASURED: no evaluation data yet

**Data needed:** >=50 subjects x {bona fide, printed photo, phone-screen replay, laptop video replay}; landmark series JSON per attempt.
- No manifest at eval/data/liveness/manifest.jsonl. Not measured.

## speaker_verification

E3 speaker verification EER and spoof acceptance, by accent.

**Status:** NOT MEASURED: no evaluation data yet

**Data needed:** >=100 speakers (mostly Indian-English), 3 sessions, >=2 devices; replay and TTS-clone probes made only from each consenting speaker's own voice.
- No manifest at eval/data/speaker_verification/manifest.jsonl. Not measured.

## lipsync

E4 lip-sync verification: AUC / EER separating genuine from mismatched clips.

**Status:** NOT MEASURED: no evaluation data yet

**Data needed:** >=100 subjects x 5 genuine 10-20 s clips (16 kHz WAV plus per-frame face landmarks JSON); constructed negatives (other-speaker audio, +/-100-500 ms offsets, playback while silent) labelled with mismatch_kind.
- No manifest at eval/data/lipsync/manifest.jsonl. Not measured.

## gaze

E5 gaze: off-screen precision / recall and Cohen's kappa vs human annotation.

**Status:** NOT MEASURED: no evaluation data yet

**Data needed:** >=30 subjects x 2 min interview-style video, per-frame landmarks plus on/off-screen labels from 2 annotators (disagreements removed or adjudicated).
- No manifest at eval/data/gaze/manifest.jsonl. Not measured.

## asr

ASR corpus WER, overall and per accent.

**Status:** NOT MEASURED: no evaluation data yet

**Data needed:** >=2 h of interview-style answers from >=50 speakers with human reference transcripts; accent subgroup per speaker.
- No manifest at eval/data/asr/manifest.jsonl. Not measured.

## question_relevance

E6 question relevance as rated by experts, per generator version.

**Status:** NOT MEASURED: no evaluation data yet

**Data needed:** 200 generated questions from >=40 consented, de-identified resume/JD pairs, each rated 1-5 by 3 experts.
- No manifest at eval/data/question_relevance/manifest.jsonl. Not measured.

## answer_scoring

E6/E9 answer-score agreement with human raters (Spearman, QWK), by subgroup.

**Status:** NOT MEASURED: no evaluation data yet

**Data needed:** >=300 answers from >=60 consenting candidates, each rated 1-5 on the published rubric by >=3 trained raters.
- No manifest at eval/data/answer_scoring/manifest.jsonl. Not measured.

## llm_schema_validity

LLM structured-output validity before and after validation/fallback.

**Status:** NOT MEASURED: no evaluation data yet

**Data needed:** Exported LLM call log from staging or E2E runs (consent_id = 'system' for synthetic sessions).
- No manifest at eval/data/llm_schema_validity/manifest.jsonl. Not measured.

## device_detection

E7 phone / second-person detection precision and recall.

**Status:** NOT MEASURED: no evaluation data yet

**Data needed:** >=500 annotated frames from consented sessions; predictions exported from the detector.
- No manifest at eval/data/device_detection/manifest.jsonl. Not measured.

## latency

End-to-end latency p50 / p95 per stage (target: first avatar frame p95 < 1.5 s).

**Status:** NOT MEASURED: no evaluation data yet

**Data needed:** Latency events exported from instrumented sessions (API /metrics/latency export).
- No manifest at eval/data/latency/manifest.jsonl. Not measured.

## avatar_sync

Avatar lip-sync quality (LSE-D lower is better, LSE-C higher is better).

**Status:** NOT MEASURED: no evaluation data yet

**Data needed:** 100 generated clips per avatar path scored with the SyncNet evaluation script (weights licence must be checked before use).
- No manifest at eval/data/avatar_sync/manifest.jsonl. Not measured.

## Smoke runs (synthetic data, pipeline check only)

| Suite | Ran | Systems exercised | Items |
|---|---|---|---|
| face_verification | yes | legacy_histogram | 32 |
| lipsync | yes | legacy_single_frame | 6 |
| gaze | yes | legacy_iris_horizontal | 10 |
| answer_scoring | yes | legacy_keyword_heuristic | 6 |
| llm_schema_validity | yes | smoke | 10 |
| device_detection | yes | smoke | 12 |
| latency | yes | question_to_first_avatar_frame | 20 |
