# Gaze and engagement (E5)

- **Purpose:** observable feedback such as "you looked away from the screen for 40% of answer 3".
- **Method:** horizontal and vertical iris position plus head yaw and pitch from MediaPipe Face Landmarker (Apache-2.0); optional per-user calibration; 5-frame majority smoothing. Implemented in both `interview_core.gaze` and `apps/web/lib/visionMath.ts`, which are parity-tested.
- **Framing rule:** reported only as behaviour with percentages and timestamps. Never as attention, interest, confidence, nervousness or any trait.
- **Evaluation:** `gaze` suite (off-screen precision/recall, Cohen's κ against annotators; by glasses and lighting). **Not measured.**
- **Known risks:** glasses glare, multi-monitor setups (looking at notes on another screen is legitimate practice), cultural differences in eye-contact norms. Feedback is advisory and never changes the coaching score.
