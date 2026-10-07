"""All evaluation suites, in report order."""

from eval_harness.suites import avsync, biometrics, gaze, nlp, runtime, speech

ALL = [
    biometrics.FACE,
    biometrics.LIVENESS,
    biometrics.SPEAKER,
    avsync.LIPSYNC,
    gaze.GAZE,
    speech.ASR,
    nlp.QUESTIONS,
    nlp.ANSWER,
    nlp.SCHEMA,
    runtime.DETECTION,
    runtime.LATENCY,
    runtime.AVATAR,
]
BY_NAME = {s.name: s for s in ALL}
