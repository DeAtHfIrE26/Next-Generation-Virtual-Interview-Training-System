# Evaluation data: what to collect and how

Every accuracy number in `docs/EVAL_REPORT.md` must come from data described here. Synthetic data is used only for smoke tests. This document is an engineering protocol, **not legal advice**. Have the consent text reviewed by counsel (and, if collected through VIT, by its ethics committee) before collecting anything.

## Ground rules

1. **Consent first.** Each participant signs the consent form below before any capture. Every manifest item carries a `consent_id` that points to that record. The harness refuses items without one.
2. **Data never enters git.** `eval/data/` is git-ignored. Store it in an encrypted bucket (India region) with access limited to named evaluators and logged.
3. **Pseudonymous IDs.** Use `subject_id`s such as `S0042`. The key linking IDs to people is kept separately by the data steward and destroyed when the retention period ends.
4. **Subgroups are self-reported and optional:** `gender`, `age_band` (18-24, 25-34, 35-44, 45+), `skin_tone_monk` (1-10, Monk Skin Tone scale), `accent` (for example `in-north`, `in-south`, `in-east`, `in-west`, `other`), `glasses`. They are used only for fairness breakdowns.
5. **Attack media is built only from the same participant's own consented captures.** For example, a printout of their own enrolment photo, or a TTS clone of their own voice. No third-party likenesses.
6. **Split by person.** No subject appears in both a calibration (threshold-setting) split and a reported test split.

## Directory layout

```
eval/data/
  face_verification/manifest.jsonl   + img/...
  liveness/manifest.jsonl            + series/...
  speaker_verification/manifest.jsonl + audio/...
  lipsync/manifest.jsonl             + clips/...
  gaze/manifest.jsonl                + clips/...
  asr/manifest.jsonl                 + audio/...
  question_relevance/manifest.jsonl
  answer_scoring/manifest.jsonl
  llm_schema_validity/manifest.jsonl (exported by the API)
  device_detection/manifest.jsonl    (labels + detector predictions)
  latency/manifest.jsonl             (exported by the API)
  avatar_sync/manifest.jsonl         (SyncNet script output)
```

Validate before running: `uv run python -m eval_harness validate`.

## Per-suite requirements

All items need `id` and `consent_id`. Paths are relative to the manifest's folder. Audio: 16 kHz mono 16-bit PCM WAV. Images: PNG/JPEG aligned face crops (the capture tool produces them).

| Suite | Minimum to start | Item fields |
|---|---|---|
| `face_verification` | 100 subjects (aim for 300). Per subject: enrolment 15 crops in one session, plus ≥10 probes across 2 lighting conditions, ≥2 devices, with and without glasses if applicable | `subject_id`, `role` (`enroll`/`probe`), `image`, `subgroups` |
| `liveness` | 50 subjects × {bona fide, printed photo, phone-screen replay, laptop video replay} | `subject_id`, `kind` (`bona_fide`/`print`/`screen_replay`/`video_replay`), `series` (landmark-series JSON from the challenge, format in `interview_core.face.liveness`) |
| `speaker_verification` | 100 speakers, 3 sessions on different days, ≥2 devices, ≥3 min each (prompted + free speech). Replay (played through a speaker) and TTS-clone probes of the same speaker | `subject_id`, `role`, `audio`, `kind` (`genuine`/`replay`/`tts_clone`), `subgroups.accent` |
| `lipsync` | 100 subjects × 5 genuine 10-20 s clips. Negatives are built from them: other-speaker audio, offsets of ±100/200/300/500 ms, audio played while the subject stays silent | `subject_id`, `audio`, `mouth_series` (`{"fps","width","height","landmarks":[[x,y]×478]×T}` or `"aperture":[...]`), `label` (`genuine`/`mismatch`), `mismatch_kind` |
| `gaze` | 30 subjects × 2 min. Two annotators label each sampled frame on-screen or off-screen; disagreements adjudicated | `subject_id`, `frames` (`{"width","height","frames":[{"landmarks","label_on_screen"}]}`) |
| `asr` | ≥2 h, ≥50 speakers, interview-style answers, human transcripts (verbatim, fillers kept) | `speaker_id`, `audio`, `reference_text`, `subgroups.accent` |
| `question_relevance` | 200 questions from ≥40 consented, de-identified resume/JD pairs, 3 expert raters, 1-5 scale | `generator` (version tag), `question`, `ratings` [r1,r2,r3] |
| `answer_scoring` | 300 answers from ≥60 candidates, ≥3 trained raters on the rubric (`docs/RUBRIC.md`) | `question`, `answer`, `role`, `human_scores`, `subgroups` |
| `device_detection` | 500 frames from consented sessions: phone fully, partly, or not visible; second person present or not | `detector`, `event` (`phone`/`second_person`), `truth`, `predicted` |

Public sets may be added as **secondary sanity checks only**, after a licence check (for example VoxCeleb1 test (CC BY 4.0) for speaker verification, AI4Bharat Svarah for Indian-English ASR). Never use LRS2/LRS3 or other non-commercial sets for anything that ships.

## Calibration

Operating thresholds (`FACE_MATCH_THRESHOLD`, `VOICE_MATCH_THRESHOLD`, the lip-sync threshold) are chosen on a **calibration split** (for example 30% of subjects) to hit the target FAR, then reported on the held-out test split. Thresholds and the split seed are recorded in `eval/thresholds.json`, and the run uses `--config eval/thresholds.json`.

## Consent form (draft for counsel review)

> **Research participation: AI Interview Coach evaluation**
>
> **Who:** [Data fiduciary legal name, address, contact email of grievance officer].
>
> **What we collect:** photos and short videos of your face, recordings of your voice, transcripts of your spoken answers, and optional self-described characteristics (age band, gender, skin tone on the Monk scale, accent, whether you wear glasses).
>
> **Why:** only to measure and improve the accuracy and fairness of identity checks, speech recognition and feedback in the AI Interview Coach. We will not use your data to train models unless you tick the box below, will not sell it, and will not use it to make decisions about you.
>
> **Attack samples:** with your permission we will create imitation samples from your own data (a printed photo, a screen replay, a synthetic copy of your voice) to test spoof detection. They are stored and deleted with your other data.
>
> **Storage and retention:** encrypted, in India, accessible only to named evaluators. Deleted within [N] months of collection or sooner on request. Aggregate statistics (for example an error rate across all participants) may be kept and published because they cannot identify you.
>
> **Your rights:** you may withdraw at any time, without giving a reason, by emailing [address]. We will delete your data within [30] days. You may ask what we hold about you and correct it.
>
> **Voluntary:** taking part is optional, and refusing has no consequences for you.
>
> ☐ I agree to the collection and use described above.
> ☐ I agree to the creation of imitation (attack) samples from my data.
> ☐ (Optional) I agree that my data may be used to **train** models, not only to evaluate them.
> ☐ (Optional) I agree to provide self-described characteristics for fairness analysis.
>
> Name / signature / date / consent ID: ______
