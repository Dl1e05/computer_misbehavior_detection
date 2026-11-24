"""Шаг 11, пакет B: оставленный предмет и «забрал другой» — сценарии O1–O8 из docs/shooting_plan.md
на синтетике: люди — цветные прямоугольники (цвет одежды нужен, чтобы узнать владельца), предмет — рамка."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from vision.analytics.tracker import Track
from vision.events.builder import event_from_detection
from vision.events.schema import parse
from vision.rules import RULES, RuleContext

FPS, W, H = 8, 1000, 1000
T0 = datetime(2026, 11, 25, 12, 0, tzinfo=timezone.utc)
NAMES = {24: "backpack", 26: "handbag", 28: "suitcase", 63: "laptop"}
RED, BLUE, GREEN = (40, 40, 220), (220, 60, 40), (40, 200, 40)
BAG_AT = (470, 640, 530, 700)                                 # где стоит сумка


def person(x, color, pid):
    """Человек ростом 300 с ногами на y=700, по центру x."""
    return (pid, (x - 50, 400, x + 50, 700), color)


def bag_with(x):
    """Сумка в руке человека, стоящего в x."""
    return (x + 20, 600, x + 80, 660)


class Scene:
    def __init__(self, **params):
        p = {"abandon_s": 10, "attend_s": 5, "confirm_s": 2, **params}
        self.rules = {k: RULES[k]("cam", p) for k in ("abandoned_object", "object_taken")}
        self.i = 0
        self.events: list = []

    def step(self, persons=(), bag=None, seconds=1.0 / FPS, bag_id=1):
        for _ in range(max(1, round(seconds * FPS))):
            img = np.full((H, W, 3), 128, np.uint8)
            for _, (x1, y1, x2, y2), color in persons:
                img[int(y1):int(y2), int(x1):int(x2)] = color
            tracks = [Track(pid, b, 0.9, 0) for pid, b, _ in persons]
            objs = [Track(bag_id, bag, 0.7, 24)] if bag else []
            ctx = RuleContext("cam", T0 + timedelta(seconds=self.i / FPS), W, H, tracks, image=img,
                              objects=objs, object_names=NAMES)
            for r in self.rules.values():
                self.events += r.update(ctx)
            self.i += 1
        return self

    def walk(self, x0, x1, color, pid, seconds, carry=False, extra=()):
        n = max(1, round(seconds * FPS))
        for k in range(n):
            x = x0 + (x1 - x0) * k / max(n - 1, 1)
            self.step([person(x, color, pid), *extra], bag_with(x) if carry else (BAG_AT if not carry else None))
        return self

    def kinds(self):
        return [e.incident_type for e in self.events]


def put_down_and_leave(s: Scene):
    """P1 (красный) приносит сумку, ставит, стоит 3 с, уходит из кадра."""
    s.walk(100, 440, RED, 1, 2, carry=True)
    s.step([person(440, RED, 1)], BAG_AT, seconds=3)
    for k in range(16):                                        # уходит вправо за 2 с
        s.step([person(440 + k * 40, RED, 1)], BAG_AT)
    return s


def test_o1_abandoned_after_owner_left():
    s = put_down_and_leave(Scene())
    s.step([], BAG_AT, seconds=12)
    assert s.kinds() == ["abandoned_object"]
    e = parse("event", event_from_detection(s.events[0], "run", "m").to_bytes())
    assert e.schema_version == "1.1" and e.bbox == (0.47, 0.64, 0.53, 0.7)
    assert e.details.object_class == "backpack" and e.details.unattended_s >= 10
    assert e.participants == []                                # владелец ушёл из кадра


def test_o1_owner_still_visible_far_is_participant():
    s = Scene()
    s.walk(100, 440, RED, 1, 2, carry=True)
    s.step([person(440, RED, 1)], BAG_AT, seconds=3)
    s.step([person(900, RED, 1)], BAG_AT, seconds=12)          # стоит далеко, но в кадре
    assert s.kinds() == ["abandoned_object"] and s.events[0].participants[0].track_id == 1


def test_o2_owner_returns_and_takes_back():
    s = put_down_and_leave(Scene())
    s.step([], BAG_AT, seconds=5)
    s.walk(900, 440, RED, 1, 2)                                # вернулся тот же трек
    s.step([person(440, RED, 1)], BAG_AT, seconds=1)
    s.walk(440, 100, RED, 1, 2, carry=True)
    assert s.kinds() == []


def test_o3_other_person_takes_it():
    s = put_down_and_leave(Scene(abandon_s=30))
    s.step([], BAG_AT, seconds=5)
    s.walk(900, 440, BLUE, 2, 2)                               # чужой подошёл
    s.step([person(440, BLUE, 2)], BAG_AT, seconds=1)
    s.walk(440, 100, BLUE, 2, 2, carry=True)                   # и унёс
    assert s.kinds() == ["object_taken"]
    e = parse("event", event_from_detection(s.events[0], "run", "m").to_bytes())
    assert e.participants[0].track_id == 2 and e.details.object_class == "backpack" and e.bbox


def test_o4_handover_in_owner_presence():
    s = Scene()
    s.walk(100, 440, RED, 1, 2, carry=True)
    s.step([person(440, RED, 1)], BAG_AT, seconds=3)
    s.step([person(440, RED, 1), person(560, BLUE, 2)], BAG_AT, seconds=1)
    for k in range(16):                                        # синий уносит, красный стоит рядом
        x = 560 - k * 25
        s.step([person(440, RED, 1), person(x, BLUE, 2)], bag_with(x))
    assert "object_taken" not in s.kinds()


def test_o5_bag_at_feet_of_seated_owner():
    s = Scene()
    s.walk(100, 440, RED, 1, 2, carry=True)
    s.step([person(440, RED, 1)], BAG_AT, seconds=40)
    assert s.kinds() == []


def test_owner_track_id_switch_is_not_abandonment():
    s = Scene()
    s.walk(100, 440, RED, 1, 2, carry=True)
    s.step([person(440, RED, 1)], BAG_AT, seconds=3)
    s.step([person(440, RED, 7)], BAG_AT, seconds=20)          # трекер сменил ID, человек тот же
    assert s.kinds() == []


def test_o6_owner_returns_with_new_track_id():
    s = put_down_and_leave(Scene(abandon_s=30))
    s.step([], BAG_AT, seconds=5)
    s.walk(900, 440, RED, 9, 2)                                # вернулся, но ID новый — узнаём по одежде
    s.step([person(440, RED, 9)], BAG_AT, seconds=1)
    s.walk(440, 100, RED, 9, 2, carry=True)
    assert s.kinds() == []


def test_passer_occluding_bag_is_not_theft():
    s = put_down_and_leave(Scene(abandon_s=60))
    s.step([], BAG_AT, seconds=3)
    s.walk(900, 500, GREEN, 3, 1)
    s.step([person(500, GREEN, 3)], None, seconds=3)           # стоит перед сумкой — её не видно
    for k in range(10):
        s.step([person(500 - k * 40, GREEN, 3)], None if k < 2 else BAG_AT)
    s.step([], BAG_AT, seconds=4)                              # сумка снова на месте
    assert s.kinds() == []


def test_bag_vanished_with_stranger_is_taken():
    s = put_down_and_leave(Scene(abandon_s=60))
    s.step([], BAG_AT, seconds=3)
    s.walk(900, 440, BLUE, 2, 2)
    s.step([person(440, BLUE, 2)], BAG_AT, seconds=1)
    for k in range(24):                                        # сумку не видно (спрятал), уходит
        s.step([person(440 - k * 15, BLUE, 2)], None)
    assert s.kinds() == ["object_taken"] and s.events[0].participants[0].track_id == 2


def test_carried_bag_never_alerts():
    s = Scene()
    s.walk(100, 900, RED, 1, 6, carry=True)                    # прошёл с сумкой через кадр
    assert s.kinds() == []


def test_scene_objects_present_from_start_are_not_abandoned():
    """Ноутбук на столе / стул, принятый за сумку, — с самого начала в кадре, никто их не приносил."""
    s = Scene()
    s.step([person(900, RED, 1)], BAG_AT, seconds=40)          # человек далеко, предмет стоит с начала
    s.walk(900, 440, BLUE, 2, 2)
    s.walk(440, 100, BLUE, 2, 2, carry=True)                   # кто-то взял «ничей» предмет
    assert s.kinds() == []


@pytest.mark.parametrize("cls,ok", [(24, True), (0, False)])
def test_only_configured_object_classes(cls, ok):
    rule = RULES["abandoned_object"]("cam", {"abandon_s": 2})
    out = []
    for i in range(8 * FPS):
        x = 100 + i * 40 if i < 8 else 500                      # 1 с несут, потом ставят и уходят
        persons = [Track(1, (x - 50, 400, x + 50, 700), 0.9, 0)] if i < 8 else []
        bag = bag_with(x) if i < 8 else BAG_AT
        ctx = RuleContext("cam", T0 + timedelta(seconds=i / FPS), W, H, persons,
                          objects=[Track(1, bag, 0.7, cls)], object_names=NAMES)
        out += rule.update(ctx)
    assert bool(out) == ok
