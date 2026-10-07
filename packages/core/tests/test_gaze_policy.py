import numpy as np
from interview_core.gaze import GazeCalibration, GazeTracker, extract_features, is_on_screen, summarise
from interview_core.security import EventType, IntegrityMonitor, Mode, PolicyConfig


def face(iris_dx=0.0, iris_dy=0.0, nose_dx=0.0):
    lm = np.full((478, 2), 0.5)
    lm[33], lm[133], lm[263], lm[362] = (0.40, 0.40), (0.46, 0.40), (0.60, 0.40), (0.54, 0.40)
    lm[159], lm[145], lm[386], lm[374] = (0.43, 0.39), (0.43, 0.41), (0.57, 0.39), (0.57, 0.41)
    lm[468:473] = (0.43 + iris_dx, 0.40 + iris_dy)
    lm[473:478] = (0.57 - iris_dx, 0.40 + iris_dy)
    lm[1] = (0.50 + nose_dx, 0.48)
    return lm


def test_features_and_uncalibrated_decision():
    f = extract_features(face())
    assert abs(f.horizontal - 0.5) < 0.01 and abs(f.vertical - 0.5) < 0.01 and abs(f.yaw) < 0.01
    assert is_on_screen(f)
    assert not is_on_screen(extract_features(face(iris_dx=0.02)))  # eyes to the side
    assert not is_on_screen(extract_features(face(iris_dy=0.009)))  # looking down at a desk
    assert not is_on_screen(extract_features(face(nose_dx=0.08)))  # head turned


def test_calibration_adapts_to_user_baseline():
    rng = np.random.default_rng(0)
    # This user's natural "looking at screen" pose has the head turned slightly.
    samples = [extract_features(face(nose_dx=0.06) + rng.normal(0, 0.0005, (478, 2))) for _ in range(30)]
    cal = GazeCalibration.fit(samples)
    assert not is_on_screen(samples[0])  # fixed limits would call this "away"
    assert is_on_screen(samples[0], cal)


def test_tracker_smooths_single_frame_flicker():
    tr = GazeTracker(window=5)
    seq = [face()] * 4 + [face(iris_dx=0.02)] + [face()] * 2
    assert all(tr.update(lm) for lm in seq)
    assert tr.update(None) is None


def test_summarise_counts_observable_behaviour():
    samples = (
        [(i * 0.1, True) for i in range(10)]
        + [(1.0 + i * 0.1, False) for i in range(10)]
        + [(2.0 + i * 0.1, None) for i in range(5)]
        + [(2.5 + i * 0.1, True) for i in range(6)]
    )
    s = summarise(samples)
    assert s["look_aways"] == 1
    assert s["longest_off_screen_s"] == 1.0
    assert 0.38 < s["off_screen_fraction"] < 0.42


def test_phone_needs_three_consecutive_frames_and_one_notice_per_episode():
    m = IntegrityMonitor()
    assert m.observe(EventType.PHONE, True, 0) is None
    assert m.observe(EventType.PHONE, True, 0.1) is None
    n = m.observe(EventType.PHONE, True, 0.2)
    assert n and n.episode == 1 and not n.end_session
    assert m.observe(EventType.PHONE, True, 0.3) is None  # same episode
    m.observe(EventType.PHONE, False, 0.4)
    for t in (1, 1.1):
        m.observe(EventType.PHONE, True, t)
    assert m.observe(EventType.PHONE, True, 1.2).episode == 2


def test_coaching_never_ends_proctored_ends_per_type():
    coach = IntegrityMonitor()
    assert not any(coach.single(EventType.VOICE_MISMATCH, t).end_session for t in range(10))
    proc = IntegrityMonitor(PolicyConfig(mode=Mode.PROCTORED))
    # Unrelated events no longer add up (prototype defect): noise + lip-sync don't end it.
    for t in range(4):
        proc.observe(EventType.BACKGROUND_NOISE, True, t)
        proc.observe(EventType.BACKGROUND_NOISE, True, t + 0.5)
        proc.observe(EventType.BACKGROUND_NOISE, False, t + 0.9)
    assert proc.single(EventType.LIPSYNC_MISMATCH, 5).end_session is False
    assert proc.single(EventType.VOICE_MISMATCH, 6).end_session is False
    assert proc.single(EventType.VOICE_MISMATCH, 7).end_session is False
    assert proc.single(EventType.VOICE_MISMATCH, 8).end_session is True
    assert proc.summary()["voice_mismatch"] == 3
