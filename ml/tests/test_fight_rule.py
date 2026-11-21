"""Драка на синтетических треках и скелетах — без видео и GPU."""
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from vision.analytics.fight_scorer import HeuristicFightScorer, find_groups, make_scorer
from vision.analytics.tracker import Track
from vision.rules import RULES, RuleContext, required_inputs

FPS = 8
T0 = datetime(2026, 11, 21, 12, 0, tzinfo=timezone.utc)
H = 300.0


def person(cx, arm_dx=0.0, arm_dy=0.0):
    """Рамка и скелет стоящего человека; руки смещены на (arm_dx, arm_dy) пикселей."""
    k = np.zeros((17, 3), np.float32)
    k[:, 2] = 0.9
    k[:, 0], k[:, 1] = cx, 500
    for i, side in ((7, -1), (8, 1)):                 # локти
        k[i, :2] = (cx + side * 40 + arm_dx * 0.5, 450 + arm_dy * 0.5)
    for i, side in ((9, -1), (10, 1)):                # запястья
        k[i, :2] = (cx + side * 50 + arm_dx, 480 + arm_dy)
    return (cx - 50, 700 - H, cx + 50, 700), k


def scene(frames, **params):
    """frames: список кадров, кадр = {track_id: (cx, arm_dx, arm_dy)}. Возвращает события и правило."""
    r = RULES["fight"]("cam-01", params)
    events = []
    for i, people in enumerate(frames):
        tracks, poses = [], {}
        for tid, (cx, dx, dy) in people.items():
            box, k = person(cx, dx, dy)
            tracks.append(Track(tid, box, 0.9, 0))
            poses[tid] = k
        events += r.update(RuleContext("cam-01", T0 + timedelta(seconds=i / FPS), 1920, 1080,
                                       tracks, poses=poses))
    return events, r


def swing(i, amp=120):
    """Руки резко ходят туда-обратно каждый кадр (удары)."""
    return amp if i % 2 else -amp


def fighting(seconds, a=(500, 1), b=(600, 2)):
    n = round(seconds * FPS)
    return [{a[1]: (a[0], swing(i), 0), b[1]: (b[0], -swing(i), 0)} for i in range(n)]


def calm(seconds, a=(500, 1), b=(600, 2)):
    return [{a[1]: (a[0], 0, 0), b[1]: (b[0], 0, 0)} for _ in range(round(seconds * FPS))]


def test_groups_by_proximity():
    boxes = {1: (0, 0, 100, 300), 2: (150, 0, 250, 300), 3: (2000, 0, 2100, 300)}
    assert find_groups(boxes) == [[1, 2]]


def test_fight_gives_suspicion_then_confirmation_same_episode():
    ev, _ = scene(fighting(5))
    kinds = [e.event_type for e in ev]
    assert kinds == ["suspicion", "confirmation"]
    assert ev[0].episode_id == ev[1].episode_id
    assert ev[0].occurred_at < ev[1].occurred_at
    assert {p.track_id for p in ev[0].participants} == {1, 2}
    assert all(e.incident_type == "fight" for e in ev)


def test_standing_close_is_not_fight():
    assert scene(calm(10))[0] == []


def test_waving_alone_is_not_fight():
    frames = [{1: (500, swing(i), 0), 2: (1500, 0, 0)} for i in range(80)]   # второй далеко
    assert scene(frames)[0] == []


def test_two_far_apart_waving_is_not_fight():
    frames = [{1: (300, swing(i), 0), 2: (1500, swing(i), 0)} for i in range(80)]
    assert scene(frames)[0] == []


def test_slow_gestures_are_not_fight():
    # разговаривают и плавно жестикулируют: руки двигаются, но медленно
    frames = [{1: (500, 20 * np.sin(i / 4), 0), 2: (600, 20 * np.cos(i / 4), 0)} for i in range(80)]
    assert scene(frames)[0] == []


def test_short_scuffle_only_suspicion():
    # короткая толкотня: подозрение есть, подтверждения нет — дальше решает BE (10 с)
    ev, _ = scene(fighting(1.5) + calm(6))
    assert [e.event_type for e in ev] == ["suspicion"]


def test_new_episode_after_calm_period():
    ev, _ = scene(fighting(4) + calm(6) + fighting(4))
    assert [e.event_type for e in ev] == ["suspicion", "confirmation"] * 2
    assert ev[0].episode_id != ev[2].episode_id


def test_one_suspicion_per_episode_during_long_fight():
    ev, _ = scene(fighting(20))
    assert [e.event_type for e in ev] == ["suspicion", "confirmation"]


def test_two_separate_fights_two_episodes():
    frames = [{**f1, **f2} for f1, f2 in zip(fighting(5), fighting(5, a=(1400, 3), b=(1500, 4)))]
    ev, _ = scene(frames)
    assert len({e.episode_id for e in ev}) == 2
    assert sorted(e.event_type for e in ev) == ["confirmation", "confirmation", "suspicion", "suspicion"]


def test_thresholds_in_seconds():
    global FPS
    for fps in (4, 16):
        old, FPS = FPS, fps
        try:
            assert [e.event_type for e in scene(fighting(5))[0]] == ["suspicion", "confirmation"]
        finally:
            FPS = old


def test_phase_and_reset():
    _, r = scene(fighting(5))
    assert r.phase_of(1) == "confirmation"
    r.reset()
    assert r.phase_of(1) is None


def test_scorer_is_pluggable():
    assert isinstance(make_scorer("heuristic"), HeuristicFightScorer)
    with pytest.raises(ValueError):
        make_scorer("video_model")
    assert required_inputs([RULES["fight"]("cam-01", {})]) == {"persons", "poses"}


def test_walking_together_toward_camera_is_not_fight():
    # двое идут рядом: всё тело (и руки вместе с ним) быстро смещается и растёт — это не драка
    r = RULES["fight"]("cam-01", {})
    ev = []
    for i in range(80):
        tracks, poses = [], {}
        for tid, cx in ((1, 500), (2, 600)):
            box, k = person(cx + i * 40, 0, 0)            # 40 px/кадр = ~1 рост/с вбок
            scale = 1 + i * 0.02                           # приближается к камере
            k = k.copy()
            k[:, :2] = (k[:, :2] - [cx, 700]) * scale + [cx + i * 40, 700]
            x1, y1, x2, y2 = box
            box = (cx + i * 40 - 50 * scale, 700 - H * scale, cx + i * 40 + 50 * scale, 700)
            tracks.append(Track(tid, box, 0.9, 0))
            poses[tid] = k
        ev += r.update(RuleContext("cam-01", T0 + timedelta(seconds=i / FPS), 4000, 1080,
                                   tracks, poses=poses))
    assert ev == []
