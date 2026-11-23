"""Огонь: фильтр M из N и блики — на мок-детекторе, без весов и GPU."""
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from vision.inference.fire import FireDetector, MockFireDetector
from vision.rules import RULES, RuleContext, build_rules, required_inputs

T0 = datetime(2026, 11, 21, 12, 0, tzinfo=timezone.utc)
W, H, HZ = 1000, 1000, 3                      # модель огня 3 раза в секунду
IMG = np.zeros((H, W, 3), np.uint8)

FIRE = (400, 400, 500, 520, 0.8, 1)           # очаг
FIRE_JIT = lambda i: (400 + i % 3 * 5, 400, 500 + i % 2 * 15, 520 + i % 3 * 10, 0.8, 1)   # дрожит
SMOKE = (300, 100, 600, 380, 0.7, 0)
ELSEWHERE = (50, 50, 120, 150, 0.8, 1)


def run(script, every_frame=1, image_fn=None, **params):
    """script — список прогонов модели; модель «запускается» на каждом every_frame-м кадре."""
    r = RULES["fire"]("cam-01", params)
    det = MockFireDetector(script)
    events, n = [], len(script) * every_frame
    for i in range(n):
        fire = det([IMG])[0] if i % every_frame == 0 else None
        img = image_fn(i // every_frame) if image_fn else IMG
        events += r.update(RuleContext("cam-01", T0 + timedelta(seconds=i / (HZ * every_frame)), W, H, [],
                                       image=img, fire=fire, fire_names={0: "smoke", 1: "fire"}))
    return events


def test_four_of_five_gives_event_with_bbox_no_participants():
    ev = run([[FIRE], [FIRE], [], [FIRE], [FIRE]])
    assert len(ev) == 1
    e = ev[0]
    assert e.incident_type == "fire" and e.event_type == "detection"
    assert e.participants == []
    assert e.bbox == (0.4, 0.4, 0.5, 0.52)


def test_three_of_five_no_event():
    assert run([[FIRE], [], [FIRE], [], [FIRE]]) == []


def test_flashes_in_different_places_no_event():
    assert run([[FIRE], [ELSEWHERE], [FIRE], [ELSEWHERE], [FIRE], [ELSEWHERE]]) == []


def test_long_fire_one_event():
    assert len(run([[FIRE]] * 60)) == 1


def test_fire_again_after_clear_gives_second_event():
    script = [[FIRE]] * 10 + [[]] * (HZ * 11) + [[FIRE]] * 10
    assert len(run(script, clear_s=10)) == 2


def test_short_gap_does_not_repeat_event():
    script = [[FIRE]] * 10 + [[]] * (HZ * 3) + [[FIRE]] * 10
    assert len(run(script, clear_s=10)) == 1


def test_frames_without_model_run_are_ignored():
    # модель гоняется каждый 3-й кадр: пропуски между прогонами не считаются «нет огня»
    assert len(run([[FIRE]] * 5, every_frame=3)) == 1


def test_smoke_only_if_enabled():
    assert run([[SMOKE]] * 6) == []
    assert len(run([[SMOKE]] * 6, classes=["fire", "smoke"])) == 1


def test_low_conf_ignored():
    weak = (400, 400, 500, 520, 0.1, 1)
    assert run([[weak]] * 6) == []


def test_flicker_filter_drops_static_lamp():
    # лампа: рамка и яркость не меняются
    assert run([[FIRE]] * 6, flicker=True) == []


def test_flicker_filter_keeps_flame():
    def img(i):
        a = np.zeros((H, W, 3), np.uint8)
        a[400:520, 400:520] = 120 + (i % 2) * 80          # яркость пламени скачет
        return a
    assert len(run([[FIRE_JIT(i)] for i in range(6)], image_fn=img, flicker=True)) == 1


def test_reset():
    r = RULES["fire"]("cam-01", {})
    for i in range(3):
        r.update(RuleContext("cam-01", T0 + timedelta(seconds=i), W, H, [], fire=np.array([FIRE], np.float32)))
    r.reset()
    ev = r.update(RuleContext("cam-01", T0 + timedelta(seconds=4), W, H, [], fire=np.array([FIRE], np.float32)))
    assert ev == []


def test_no_weights_detector_is_unavailable_not_crash(tmp_path):
    d = FireDetector(str(tmp_path / "nope.pt"))
    assert not d.available and "нет весов" in d.reason
    out = d([IMG, IMG])
    assert len(out) == 2 and all(o.shape == (0, 6) for o in out)


def test_requires_fire():
    assert required_inputs(build_rules("cam-01", {"fire": {}})) == {"fire"}


def test_node_status_shows_disabled_rule():
    from vision.runtime.status import NodeStatus
    s = NodeStatus()
    s.set_camera("cam-01", "online")
    s.set_rule("cam-01", "fire", "off: нет весов")
    s.set_rule("cam-01", "zone_intrusion", "on")
    snap = s.snapshot()
    assert snap["cameras"][0]["rules"] == {"fire": "off: нет весов", "zone_intrusion": "on"}
    assert "fire: off" in s.describe()
