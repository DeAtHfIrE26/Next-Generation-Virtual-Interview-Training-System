from datetime import UTC, datetime, timedelta

import numpy as np
from interview_core.face.liveness import (
    Challenge,
    eye_aspect_ratio,
    issue_challenge,
    series_digest,
    verify_challenge,
    yaw_proxy,
)

EYE_L = {
    33: (0.40, 0.40),
    160: (0.42, 0.39),
    158: (0.44, 0.39),
    133: (0.46, 0.40),
    153: (0.44, 0.41),
    144: (0.42, 0.41),
}
EYE_R = {
    362: (0.54, 0.40),
    385: (0.56, 0.39),
    387: (0.58, 0.39),
    263: (0.60, 0.40),
    373: (0.58, 0.41),
    380: (0.56, 0.41),
}
LIPS = {
    13: (0.50, 0.55),
    14: (0.50, 0.56),
    82: (0.48, 0.55),
    87: (0.48, 0.56),
    312: (0.52, 0.55),
    317: (0.52, 0.56),
}


def face(pose="neutral"):
    lm = np.full((478, 2), 0.5)
    for d in (EYE_L, EYE_R, LIPS):
        for k, v in d.items():
            lm[k] = v
    lm[1] = (0.50, 0.48)
    if pose == "blink":
        for k in (160, 158, 153, 144, 385, 387, 373, 380):
            lm[k, 1] = 0.40
    elif pose == "turn_left":
        lm[1, 0] += 0.04
    elif pose == "turn_right":
        lm[1, 0] -= 0.04
    elif pose == "open_mouth":
        for k in (14, 87, 317):
            lm[k, 1] += 0.06
    return lm.tolist()


def performance(nonce, steps, fps=15, jitter=0.0, seed=0):
    rng = np.random.default_rng(seed)
    poses = ["neutral"] * fps
    for s in steps:
        poses += [s] * int(0.3 * fps) + ["neutral"] * int(0.6 * fps)
    frames = []
    for i, p in enumerate(poses):
        lm = np.asarray(face(p)) + rng.normal(0, jitter, (478, 2))
        frames.append({"t": i / fps, "landmarks": lm.tolist(), "faces": 1})
    return {"nonce": nonce, "frames": frames}


def _challenge(steps=("blink", "turn_left", "open_mouth")):
    now = datetime.now(UTC)
    return Challenge("abc", steps, now, now + timedelta(seconds=30))


def test_features():
    assert eye_aspect_ratio(np.asarray(face())) > 0.3
    assert eye_aspect_ratio(np.asarray(face("blink"))) < 0.05
    assert yaw_proxy(np.asarray(face("turn_left"))) > 0.15 > yaw_proxy(np.asarray(face()))


def test_correct_performance_passes_with_landmark_jitter():
    ch = _challenge()
    r = verify_challenge(ch, performance("abc", ch.steps, jitter=0.0005))
    assert r.passed, r.reasons
    assert [d["step"] for d in r.detected] == list(ch.steps)


def test_wrong_order_and_static_photo_fail():
    ch = _challenge()
    assert not verify_challenge(ch, performance("abc", ("open_mouth", "turn_left", "blink"))).passed
    photo = {"nonce": "abc", "frames": [{"t": i / 15, "landmarks": face(), "faces": 1} for i in range(60)]}
    r = verify_challenge(ch, photo)
    assert not r.passed and r.score == 0


def test_one_pose_cannot_satisfy_two_steps():
    ch = _challenge(("turn_left", "turn_left", "blink"))
    held = performance("abc", ("turn_left",))  # one turn only
    held["frames"] += performance("abc", ("blink",))["frames"][15:]
    assert not verify_challenge(ch, held).passed


def test_replay_guards():
    ch = _challenge()
    good = performance("abc", ch.steps)
    assert not verify_challenge(ch, good, already_used=True).passed
    assert not verify_challenge(ch, good, now=ch.expires_at + timedelta(seconds=1)).passed
    assert not verify_challenge(ch, {**good, "nonce": "other"}).passed
    assert series_digest(good) == series_digest(performance("abc", ch.steps))


def test_capture_quality_guards():
    ch = _challenge()
    slow = performance("abc", ch.steps, fps=5)
    assert "frame rate" in verify_challenge(ch, slow).reasons[0]
    two = performance("abc", ch.steps)
    for f in two["frames"][::3]:
        f["faces"] = 2
    assert not verify_challenge(ch, two).passed


def test_issued_challenges_are_random_and_single_purpose():
    seen = {issue_challenge().steps for _ in range(200)}
    assert len(seen) > 10
    ch = issue_challenge()
    assert len(set(ch.steps)) == 3 and ch.expires_at > ch.issued_at
