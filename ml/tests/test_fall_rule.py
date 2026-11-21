"""Правило падения на синтетических скелетах — без видео и GPU."""
from datetime import datetime, timedelta, timezone

import numpy as np

from vision.analytics.tracker import Track
from vision.inference.pose import Pose, attach_poses
from vision.rules import RULES, RuleContext, build_rules, required_inputs

FPS = 8
T0 = datetime(2026, 11, 21, 12, 0, tzinfo=timezone.utc)
H = 300.0                    # рост человека в пикселях
CX, GROUND = 500.0, 800.0    # где стоит


def standing():
    """bbox и 17 точек стоящего человека."""
    k = np.zeros((17, 3), np.float32)
    k[:, 2] = 0.9
    k[:, 0] = CX
    k[:, 1] = GROUND - H * 0.5
    k[0] = (CX, GROUND - H * 0.95, 0.9)                  # нос
    k[5], k[6] = (CX - 30, GROUND - H * 0.8, 0.9), (CX + 30, GROUND - H * 0.8, 0.9)
    k[11], k[12] = (CX - 20, GROUND - H * 0.5, 0.9), (CX + 20, GROUND - H * 0.5, 0.9)
    k[15], k[16] = (CX - 20, GROUND, 0.9), (CX + 20, GROUND, 0.9)
    return (CX - 50, GROUND - H, CX + 50, GROUND), k


def lying():
    k = np.zeros((17, 3), np.float32)
    k[:, 2] = 0.9
    k[:, 0] = CX
    k[:, 1] = GROUND - 30
    k[0] = (CX - H * 0.45, GROUND - 30, 0.9)
    k[5], k[6] = (CX - H * 0.3, GROUND - 40, 0.9), (CX - H * 0.3, GROUND - 20, 0.9)
    k[11], k[12] = (CX, GROUND - 40, 0.9), (CX, GROUND - 20, 0.9)
    k[15], k[16] = (CX + H * 0.45, GROUND - 40, 0.9), (CX + H * 0.45, GROUND - 20, 0.9)
    return (CX - H * 0.5, GROUND - 70, CX + H * 0.5, GROUND), k


def squatting():
    bbox, k = standing()
    k = k.copy()
    k[[0, 5, 6, 11, 12], 1] += H * 0.35                   # всё выше колен опустилось, торс вертикален
    return (CX - 60, GROUND - H * 0.65, CX + 60, GROUND), k


def lerp(a, b, s):
    (ba, ka), (bb, kb) = a, b
    return tuple(x + (y - x) * s for x, y in zip(ba, bb)), ka + (kb - ka) * s


def seq(*parts):
    """parts: (поза_или_функция(s), секунд). Возвращает список (bbox, kpts) по кадрам."""
    out = []
    for pose, seconds in parts:
        n = max(1, round(seconds * FPS))
        for i in range(n):
            out.append(pose(i / n) if callable(pose) else pose)
    return out


def run(frames, rule=None, jitter=0.0, **params):
    r = rule or RULES["fall"]("cam-01", params)
    rng = np.random.default_rng(0)
    events = []
    for i, (bbox, k) in enumerate(frames):
        k = k.copy()
        k[:, :2] += rng.normal(0, jitter, (17, 2))
        ctx = RuleContext("cam-01", T0 + timedelta(seconds=i / FPS), 1920, 1080,
                          [Track(1, bbox, 0.9, 0)], poses={1: k})
        events += r.update(ctx)
    return events, r


S, L, Q = standing(), lying(), squatting()


def test_fall_and_lie_gives_one_event():
    ev, r = run(seq((S, 1), (lambda s: lerp(S, L, s), 0.5), (L, 5)), jitter=1.0)
    assert len(ev) == 1
    e = ev[0]
    assert e.incident_type == "fall" and e.event_type == "detection"
    assert e.participants[0].track_id == 1
    assert r.phase_of(1) == "on_ground"


def test_event_only_after_still_seconds():
    frames = seq((S, 1), (lambda s: lerp(S, L, s), 0.5), (L, 2.5))   # лежит меньше 3 с
    assert run(frames, still_s=3)[0] == []


def test_fall_and_get_up_no_event():
    frames = seq((S, 1), (lambda s: lerp(S, L, s), 0.5), (L, 1.5),
                 (lambda s: lerp(L, S, s), 1), (S, 3))
    ev, r = run(frames)
    assert ev == [] and r.phase_of(1) == "upright"


def test_squat_no_event():
    assert run(seq((S, 1), (lambda s: lerp(S, Q, s), 0.5), (Q, 6)))[0] == []


def test_lie_down_slowly_no_event():
    # лёг на лавку за 4 с — не падение, даже если потом лежит неподвижно
    assert run(seq((S, 1), (lambda s: lerp(S, L, s), 4), (L, 6)))[0] == []


def test_moving_on_ground_waits_for_stillness():
    frames = seq((S, 1), (lambda s: lerp(S, L, s), 0.5), (L, 5))
    assert run(frames, jitter=25.0)[0] == []          # сильно шевелится — ещё не событие


def test_hidden_torso_no_event():
    hidden = []
    for bbox, k in seq((S, 1), (lambda s: lerp(S, L, s), 0.5), (L, 5)):
        k = k.copy()
        k[[5, 6, 11, 12], 2] = 0.1                    # плечи и бёдра не видны
        hidden.append((bbox, k))
    assert run(hidden)[0] == []


def test_no_pose_no_event():
    r = RULES["fall"]("cam-01", {})
    ev = []
    for i, (bbox, _) in enumerate(seq((S, 1), (lambda s: lerp(S, L, s), 0.5), (L, 5))):
        ev += r.update(RuleContext("cam-01", T0 + timedelta(seconds=i / FPS), 1920, 1080,
                                   [Track(1, bbox, 0.9, 0)], poses={}))
    assert ev == []


def test_one_event_per_fall_then_again_after_getting_up():
    fall = [(S, 1), (lambda s: lerp(S, L, s), 0.5), (L, 4)]
    frames = seq(*fall, (lambda s: lerp(L, S, s), 1), (S, 2), *fall)
    assert len(run(frames)[0]) == 2


def test_thresholds_in_seconds():
    for fps in (4, 16):
        global FPS
        old, FPS = FPS, fps
        try:
            assert len(run(seq((S, 1), (lambda s: lerp(S, L, s), 0.5), (L, 5)))[0]) == 1
        finally:
            FPS = old


def test_reset_clears_state():
    _, r = run(seq((S, 1), (lambda s: lerp(S, L, s), 0.5), (L, 1)))
    assert r.phase_of(1) == "on_ground"
    r.reset()
    assert r.phase_of(1) is None


def test_attach_poses_by_iou():
    tracks = [Track(7, (0, 0, 100, 200), 0.9, 0), Track(8, (300, 0, 400, 200), 0.9, 0)]
    k1, k2 = np.ones((17, 3)), np.full((17, 3), 2.0)
    poses = [Pose((305, 5, 400, 200), 0.9, k2), Pose((2, 0, 100, 195), 0.9, k1),
             Pose((800, 0, 900, 200), 0.9, k1)]                 # третий ни к кому
    got = attach_poses(tracks, poses)
    assert set(got) == {7, 8}
    assert got[7] is k1 and got[8] is k2


def test_required_inputs():
    rules = build_rules("cam-01", {"fall": {"still_s": 3}})
    assert required_inputs(rules) == {"persons", "poses"}


def test_pose_lost_during_fall_still_detected():
    # в момент падения скелета нет (смаз), есть только рамка детектора
    frames = []
    for i, (bbox, k) in enumerate(seq((S, 1), (lambda s: lerp(S, L, s), 0.75), (L, 5))):
        k = k.copy()
        if 8 <= i < 14:
            k[:, 2] = 0.0
        frames.append((bbox, k))
    assert len(run(frames)[0]) == 1
