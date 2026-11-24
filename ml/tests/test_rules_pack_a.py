"""Шаг 11, пакет A: линия, долгое нахождение, скопление, бег, лежит без движения, расписание зон,
саботаж камеры. Синтетические треки и картинки — без видео и GPU."""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import cv2
import numpy as np
import pytest

from vision.analytics.tracker import Track
from vision.analytics.zones import Line, Zone, load_zones, save_zones
from vision.events.builder import event_from_detection
from vision.events.schema import is_agreed, parse
from vision.rules import RULES, RuleContext, build_rules

W, H, FPS = 1000, 1000, 8
T0 = datetime(2026, 11, 27, 12, 0, tzinfo=timezone.utc)          # пятница
ZID = "0f8e7d6c-5b4a-4392-8170-6e5d4c3b2a19"                       # UUID, как из cameras-zones
ZONE = Zone(ZID, "cam", [(0.2, 0.2), (0.8, 0.2), (0.8, 0.8), (0.2, 0.8)], name="z")


def person(tid, cx, foot_y, h=200, w=80, conf=0.9):
    """Рамка человека по центру x и линии ног (пиксели)."""
    return Track(tid, (cx - w / 2, foot_y - h, cx + w / 2, foot_y), conf, 0)


def run(rule, frames, t0=T0, image=None):
    """frames: список списков треков по кадрам (8 кадр/с). -> все события."""
    out = []
    for i, tracks in enumerate(frames):
        out += rule.update(RuleContext("cam", t0 + timedelta(seconds=i / FPS), W, H, tracks, image=image))
    return out


def as_event(d):
    """Событие правила проходит контракт (schema 1.1) и сериализуется."""
    e = event_from_detection(d, "ml-run", "ml-0.1.0/test")
    return parse("event", e.to_bytes())


# ---------- расписание зон ----------

def test_schedule_night_window_belongs_to_start_day():
    z = Zone("n", "cam", ZONE.polygon, schedule=[{"days": "mon-fri", "from": "22:00", "to": "07:00"}],
             tz="Asia/Almaty")                                              # UTC+5
    at = lambda d, hh: datetime(2026, 11, d, hh, 0, tzinfo=timezone.utc) - timedelta(hours=5)  # noqa: E731
    assert z.armed(at(27, 23))          # пятница 23:00
    assert z.armed(at(28, 6))           # суббота 06:00 — окно началось в пятницу
    assert not z.armed(at(28, 23))      # суббота 23:00 — выходной
    assert not z.armed(at(30, 6))       # понедельник 06:00 — воскресенье не в days
    assert not z.armed(at(27, 12))      # пятница днём
    assert Zone("a", "cam", ZONE.polygon).armed(at(27, 12))                  # без расписания — всегда


def test_zone_intrusion_fires_when_schedule_starts_with_person_inside():
    z = Zone("night", "cam", ZONE.polygon, schedule=[{"days": "daily", "from": "17:00", "to": "07:00"}], tz="UTC")
    rule = RULES["zone_intrusion"]("cam", {"zones": [z], "min_inside_s": 0.5})
    t0 = datetime(2026, 11, 27, 16, 59, 58, tzinfo=timezone.utc)
    ev = run(rule, [[person(1, 500, 500)]] * 40, t0=t0)                     # стоит внутри с 16:59:58
    assert len(ev) == 1 and ev[0].occurred_at >= datetime(2026, 11, 27, 17, 0, tzinfo=timezone.utc)


def test_geometry_file_roundtrip_with_lines_and_timezone(tmp_path):
    path = tmp_path / "zones.json"
    lid = "1a2b3c4d-5e6f-4a7b-8c9d-0e1f2a3b4c5d"
    items = {ZID: Zone(ZID, "cam", ZONE.polygon, name="z",
                       schedule=[{"days": "sat,sun", "from": "00:00", "to": "24:00"}]),
             lid: Line(lid, "cam", [(0.1, 0.5), (0.9, 0.5)], name="l")}
    save_zones(path, items, timezone="Asia/Almaty")
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["timezone"] == "Asia/Almaty" and data["lines"][0]["line_id"] == lid and data["lines"][0]["name"] == "l"
    back = load_zones(path)
    assert isinstance(back[lid], Line) and back[ZID].tz == "Asia/Almaty"
    rules = build_rules("cam", {"line_crossing": {"lines": ["l"]}, "zone_intrusion": {}}, back)
    assert rules[0].lines.keys() == {lid} and rules[1].zones.keys() == {ZID}
    with pytest.raises(ValueError):
        build_rules("cam", {"line_crossing": {"lines": ["z"]}}, back)              # зона — не линия


# ---------- линия ----------

LID = "9e8d7c6b-5a49-4382-a716-05f4e3d2c1b0"
VLINE = Line(LID, "cam", [(0.5, 0.0), (0.5, 1.0)], name="door")     # сверху вниз: «справа» — это левая часть кадра


def walk_x(xs, tid=1, foot_y=600):
    return [[person(tid, x, foot_y)] for x in xs]


def test_line_crossing_both_directions():
    rule = RULES["line_crossing"]("cam", {"lines": [VLINE]})
    ev = run(rule, walk_x(list(range(800, 199, -40)) + [200] * 8 + list(range(200, 801, 40)) + [800] * 8))
    assert [e.extra["direction"] for e in ev] == ["in", "out"]
    e = as_event(ev[0])
    assert e.schema_version == "1.1" and str(e.line_id) == LID and e.details.direction == "in"
    assert not is_agreed("line_crossing")


def test_line_jitter_is_not_crossing():
    rule = RULES["line_crossing"]("cam", {"lines": [VLINE], "min_side_s": 0.3})
    xs = [800] * 8 + [520, 480] * 20 + [800] * 8                           # дрожит у линии по кадру
    assert run(rule, walk_x(xs)) == []


def test_line_segment_bypass_and_direction_filter():
    short = Line("2c3d4e5f-6a7b-4c8d-9e0f-1a2b3c4d5e6f", "cam", [(0.5, 0.2), (0.5, 0.4)])                     # короткий отрезок наверху
    rule = RULES["line_crossing"]("cam", {"lines": [short]})
    assert run(rule, walk_x(range(800, 199, -40), foot_y=900)) == []        # прошёл ниже отрезка
    only_out = RULES["line_crossing"]("cam", {"lines": [VLINE], "direction": "out"})
    ev = run(only_out, walk_x(list(range(800, 199, -40)) + [200] * 8 + list(range(200, 801, 40)) + [800] * 8))
    assert [e.extra["direction"] for e in ev] == ["out"]


# ---------- долгое нахождение ----------

def test_loitering_with_short_exits():
    rule = RULES["loitering"]("cam", {"zones": [ZONE], "dwell_s": 60, "gap_s": 3})
    inside, outside = [person(1, 500, 500)], [person(1, 900, 950)]
    frames = [inside] * (30 * FPS) + [outside] * FPS + [inside] * (35 * FPS)   # выходил на 1 с
    ev = run(rule, frames)
    assert len(ev) == 1 and 60 <= ev[0].extra["dwell_s"] < 62
    e = as_event(ev[0])
    assert str(e.zone_id) == ZID and e.details.dwell_s >= 60


def test_loitering_resets_after_long_exit():
    rule = RULES["loitering"]("cam", {"zones": [ZONE], "dwell_s": 10, "gap_s": 3})
    inside, outside = [person(1, 500, 500)], [person(1, 900, 950)]
    ev = run(rule, [inside] * (8 * FPS) + [outside] * (5 * FPS) + [inside] * (8 * FPS))
    assert ev == []                                                         # 8 + 8 с, но с долгим выходом


# ---------- скопление ----------

def crowd(n):
    return [person(i, 250 + (i % 5) * 120, 300 + (i // 5) * 200) for i in range(n)]


def test_crowding_once_and_rearm():
    rule = RULES["crowding"]("cam", {"zones": [ZONE], "min_count": 8, "min_s": 5, "clear_s": 10})
    frames = [crowd(9)] * (7 * FPS) + [crowd(3)] * (11 * FPS) + [crowd(9)] * (6 * FPS)
    ev = run(rule, frames)
    assert len(ev) == 2 and ev[0].extra["count"] == 9
    assert as_event(ev[0]).details.count == 9


def test_crowding_short_spike_ignored():
    rule = RULES["crowding"]("cam", {"zones": [ZONE], "min_count": 8, "min_s": 5})
    assert run(rule, [crowd(3)] * 16 + [crowd(10)] * (3 * FPS) + [crowd(3)] * 16) == []


# ---------- бег ----------

def move(speed_heights, seconds, h=200):
    """Ровное движение со скоростью в ростах/с."""
    step = speed_heights * h / FPS
    return [[person(1, 100 + i * step, 700, h=h)] for i in range(int(seconds * FPS))]


def test_running_vs_walking():
    assert run(RULES["running"]("cam", {}), move(0.8, 6)) == []             # шаг
    ev = run(RULES["running"]("cam", {}), move(2.5, 4))                    # бег
    assert len(ev) == 1 and ev[0].extra["speed"] >= 2.0
    assert as_event(ev[0]).details.speed >= 2.0


def test_running_single_jump_ignored():
    frames = [[person(1, 100, 700)]] * 20 + [[person(1, 900, 700)]] * 20   # рывок рамки при перекрытии
    assert run(RULES["running"]("cam", {"min_s": 1.0}), frames) == []


def test_running_scale_free_far_person():
    assert len(run(RULES["running"]("cam", {}), move(2.5, 4, h=60))) == 1   # маленький, но бежит


# ---------- лежит без движения ----------

def lying(tid=1, dx=0.0):
    return Track(tid, (300 + dx, 600, 600 + dx, 700), 0.9, 0)               # шире, чем выше


def test_lying_motionless_without_pose_lower_confidence():
    rule = RULES["lying_motionless"]("cam", {"lie_s": 30})
    ev = run(rule, [[lying()]] * (31 * FPS))
    assert len(ev) == 1 and ev[0].confidence == pytest.approx(0.63)        # 0.9 × 0.7 — без скелета
    assert as_event(ev[0]).incident_type == "lying_motionless"


def test_lying_but_moving_or_standing_up():
    rule = RULES["lying_motionless"]("cam", {"lie_s": 10})
    wriggle = [[lying(dx=(30 if i % 2 else 0))] for i in range(15 * FPS)]   # ворочается
    assert run(rule, wriggle) == []
    rule = RULES["lying_motionless"]("cam", {"lie_s": 10})
    frames = [[lying()]] * (8 * FPS) + [[person(1, 450, 700)]] * FPS + [[lying()]] * (8 * FPS)
    assert run(rule, frames) == []                                          # встал на 1 с — отсчёт заново


# ---------- саботаж камеры ----------

def scene(seed=0):
    rng = np.random.default_rng(seed)
    img = cv2.resize(rng.integers(0, 255, (24, 32, 3), dtype=np.uint8), (640, 480), interpolation=cv2.INTER_NEAREST)
    cv2.rectangle(img, (100, 100), (300, 400), (255, 255, 255), 8)
    cv2.line(img, (0, 240), (640, 300), (0, 0, 0), 6)
    return img


def tamper_run(rule, images, t0=T0):
    out = []
    for i, img in enumerate(images):
        out += rule.update(RuleContext("cam", t0 + timedelta(seconds=i / FPS), 640, 480, [], image=img))
    return out


@pytest.mark.parametrize("kind,make", [
    ("covered", lambda s: np.full_like(s, 40)),
    ("glare", lambda s: np.full_like(s, 255)),
    ("defocused", lambda s: cv2.GaussianBlur(s, (0, 0), 12)),
    ("moved", lambda s: np.roll(s, (120, 200), (0, 1))),                  # камеру развернули
])
def test_tampering_kinds(kind, make):
    base = scene()
    rule = RULES["camera_tampering"]("cam", {})
    noisy = [cv2.add(base, np.random.default_rng(i).integers(0, 6, base.shape, dtype=np.uint8))
             for i in range(8 * FPS)]
    assert tamper_run(rule, noisy) == []                                    # учится и молчит на норме
    bad = make(base)
    ev = tamper_run(rule, [bad] * (5 * FPS), t0=T0 + timedelta(seconds=8))
    assert [e.extra["tamper_type"] for e in ev] == [kind]
    e = as_event(ev[0])
    assert e.participants == [] and e.details.tamper_type == kind


def test_tampering_short_obstruction_and_person_ignored():
    base = scene()
    rule = RULES["camera_tampering"]("cam", {"hold_s": 3})
    tamper_run(rule, [base] * (6 * FPS))
    passer = base.copy()
    cv2.rectangle(passer, (380, 150), (470, 470), (90, 60, 40), -1)         # человек прошёл мимо
    covered = np.full_like(base, 30)
    frames = [passer] * (4 * FPS) + [covered] * FPS + [base] * (3 * FPS)    # рука на 1 с
    assert tamper_run(rule, frames, t0=T0 + timedelta(seconds=6)) == []


def test_tampering_one_event_per_episode():
    base = scene()
    rule = RULES["camera_tampering"]("cam", {"hold_s": 2, "clear_s": 3})
    tamper_run(rule, [base] * (6 * FPS))
    cov = np.full_like(base, 20)
    frames = [cov] * (10 * FPS) + [base] * (4 * FPS) + [cov] * (3 * FPS)
    ev = tamper_run(rule, frames, t0=T0 + timedelta(seconds=6))
    assert [e.extra["tamper_type"] for e in ev] == ["covered", "covered"]   # долгое — одно; после нормы — снова


def test_tampering_reference_not_learned_on_black_start():
    """Ролик/камера начинается с чёрного экрана — эталон учится только на нормальных кадрах."""
    base = scene()
    rule = RULES["camera_tampering"]("cam", {"learn_s": 1})
    frames = [np.zeros_like(base)] * (2 * FPS) + [base] * (8 * FPS)
    assert tamper_run(rule, frames) == []
