# Lip-sync verification (E4)

- **Purpose:** detect when the heard speech does not come from the visible speaker: another person speaking, played-back audio, or dubbed/offset audio.
- **Method:**
  1. Mouth aperture (3 inner-lip pairs over inter-ocular distance) is computed on-device at about 12 fps.
  2. Server-side, the series is correlated with the 300-3000 Hz speech energy envelope over ±0.5 s of lag. The peak gives the correlation and the offset.
  3. The peak must exceed circular-shift chance alignment (prominence).
  4. Flags: speech with a still mouth, offset over 200 ms, sustained mouth motion without speech.
  5. Too little speech or face coverage gives "inconclusive", not a verdict.

  Code: `interview_core.lipsync.avsync`.
- **Prototype comparison:** the prototype scored one frame and could not produce a warning. The new check fires on constructed mismatches in unit tests (synthetic signals; not an accuracy result).
- **Thresholds:** provisional defaults (`AVSyncConfig`). Calibrate on the `lipsync` suite before enforcing in proctored mode.
- **Evaluation:** `lipsync` suite (AUC/EER, per mismatch kind; offset error). **Not measured.**
- **Known risks:** beards and masks, low frame rates, latency differences between camera and mic pipelines (absorbed by the lag search up to the limit), people who speak with little mouth movement.
