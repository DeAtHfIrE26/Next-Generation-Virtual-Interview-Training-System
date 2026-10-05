# Interviewer avatar and voice (not a patent element)

- **Avatar:** original 2D vector rig (no third-party likeness or weights) driven by viseme timelines from TTS marks, browser word boundaries or a text estimate. Optional neural talking head behind `FEATURE_NEURAL_AVATAR`, which must use a commercially licensed model (`services/gpu/README.md`); any failure falls back to the rig.
- **Voice:** Amazon Polly neural voice (default en-IN "Kajal") with viseme and word marks, or browser speech synthesis.
- **Metrics:** `question_to_first_avatar_frame` latency (target p95 < 1.5 s), LSE-D/LSE-C for the rig and the neural path (`avatar_sync` suite). **Not measured.**
- **Open item:** offer several avatar appearances so the default does not stand in for everyone.
