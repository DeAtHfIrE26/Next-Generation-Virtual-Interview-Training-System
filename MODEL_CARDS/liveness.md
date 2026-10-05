# Active liveness (E2 hardening)

- **Purpose:** resist printed-photo and pre-recorded-video presentation during enrolment.
- **Method:** server-issued, single-use challenge with a random order of 3 of {blink, turn left, turn right, open mouth} and a 30 s expiry. Detected from face-landmark geometry: eye aspect ratio, nose offset over inter-ocular distance, mouth aperture. Steps must occur in order, with a return to neutral between them. Replayed series are rejected by digest. Code: `interview_core.face.liveness`.
- **Not covered:** high-quality masks, real-time deepfake puppeteering, screen replays of a live performance. A trained passive presentation-attack detector (`PassivePAD` interface) is needed for those and is **not configured**.
- **Evaluation:** `liveness` suite (APCER per attack type, BPCER). **Not measured.**
- **Accessibility:** users who cannot perform a step (for example, limited head movement) need an alternative path. **Open item.**
