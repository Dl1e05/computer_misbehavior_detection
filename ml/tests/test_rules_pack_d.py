"""Шаг 11, пакет D: изменение охраняемого объекта (сигнал вандализма) — синтетическая «дверь»."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import cv2
import numpy as np

from vision.analytics.tracker import Track
from vision.analytics.zones import Zone
from vision.events.builder import event_from_detection
from vision.events.schema import parse
from vision.rules import RULES, RuleContext

FPS, W, H = 8, 640, 480
T0 = datetime(2026, 11, 25, 12, 0, tzinfo=timezone.utc)
ZID = "6b1e2c3d-4a5f-4e6d-8c7b-9a0f1e2d3c4b"
DOOR = Zone(ZID, "cam", [(0.3, 0.2), (0.6, 0.2), (0.6, 0.6), (0.3, 0.6)], name="door")
DX1, DY1, DX2, DY2 = int(0.3 * W), int(0.2 * H), int(0.6 * W), int(0.6 * H)


def scene() -> np.ndarray:
    rng = np.random.default_rng(0)
    img = cv2.resize(rng.integers(60, 200, (30, 40, 3), dtype=np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
    door = img[DY1:DY2, DX1:DX2]
    door[:] = (120, 90, 60)                                       # дверь: панели и ручка
    for k in range(4):
        cv2.rectangle(door, (10, 10 + k * 45), (door.shape[1] - 10, 45 + k * 45), (90, 60, 40), 3)
    cv2.circle(door, (door.shape[1] - 25, door.shape[0] // 2), 8, (200, 200, 200), -1)
    return img


def graffiti(img: np.ndarray, frac: float = 1.0) -> np.ndarray:
    """Исписали/испортили: случайные яркие пятна и линии на доле frac двери."""
    out = img.copy()
    rng = np.random.default_rng(5)
    w = int((DX2 - DX1) * frac)
    patch = cv2.resize(rng.integers(0, 255, (12, max(2, int(16 * frac)), 3), dtype=np.uint8), (w, DY2 - DY1),
                       interpolation=cv2.INTER_NEAREST)
    out[DY1:DY2, DX1:DX1 + w] = patch
    return out


def person(pid: int, x: float, color=(30, 30, 200)) -> tuple[Track, tuple]:
    """Человек ростом 0.5 кадра, ноги на 0.95; x — центр в долях."""
    b = (x * W - 30, 0.45 * H, x * W + 30, 0.95 * H)
    return Track(pid, b, 0.9, 0), color


class Cam:
    def __init__(self, **params):
        self.rule = RULES["scene_change"]("cam", {"zones": [DOOR], **params})
        self.i, self.events = 0, []

    def show(self, img, people=(), seconds=1.0 / FPS):
        for _ in range(max(1, round(seconds * FPS))):
            frame = img.copy()
            tracks = []
            for t, color in people:
                x1, y1, x2, y2 = map(int, t.bbox)
                frame[y1:y2, x1:x2] = color
                tracks.append(t)
            ctx = RuleContext("cam", T0 + timedelta(seconds=self.i / FPS), W, H, tracks, image=frame)
            self.events += self.rule.update(ctx)
            self.i += 1
        return self


def vandal_visits(cam: Cam, base: np.ndarray, pid: int = 7):
    """Человек подходит к двери справа, стоит 3 с рядом и уходит."""
    for x in np.linspace(0.95, 0.7, 12):
        cam.show(base, [person(pid, x)])
    cam.show(base, [person(pid, 0.68)], seconds=3)
    for x in np.linspace(0.7, 1.05, 12):
        cam.show(base, [person(pid, x)] if x < 1.0 else [])


def test_vandalism_signal_after_person_nearby():
    base, cam = scene(), Cam()
    cam.show(base, seconds=6)                                     # учит эталон
    vandal_visits(cam, base)
    cam.show(graffiti(base), seconds=10)                          # дверь испорчена и такой остаётся
    assert len(cam.events) == 1
    d = cam.events[0]
    assert d.extra["change_ratio"] >= 0.25 and d.participants[0].track_id == 7
    e = parse("event", event_from_detection(d, "run", "m").to_bytes())
    assert e.schema_version == "1.1" and str(e.zone_id) == ZID and e.details.seconds_since_person < 5


def test_lighting_change_is_not_vandalism():
    base, cam = scene(), Cam()
    cam.show(base, seconds=6)
    vandal_visits(cam, base)
    darker = cv2.convertScaleAbs(base, alpha=0.4, beta=0)         # сильно убавили свет (без нормировки — ложное)
    cam.show(darker, seconds=12)
    assert cam.events == []


def test_change_without_person_nearby_is_ignored():
    base, cam = scene(), Cam()
    cam.show(base, seconds=6)
    cam.show(graffiti(base), seconds=12)
    assert cam.events == []


def test_person_occluding_door_is_not_change():
    base, cam = scene(), Cam()
    cam.show(base, seconds=6)
    cam.show(base, [person(3, 0.45)], seconds=20)                 # стоит прямо перед дверью
    cam.show(base, seconds=4)
    assert cam.events == []


def test_small_sticker_and_short_flicker_ignored():
    base, cam = scene(), Cam()
    cam.show(base, seconds=6)
    vandal_visits(cam, base)
    cam.show(graffiti(base, frac=0.1), seconds=10)                # наклейка на 10% двери
    cam.show(graffiti(base), seconds=2)                           # мелькнуло 2 с
    cam.show(base, seconds=4)
    assert cam.events == []


def test_one_event_per_change_then_new_change_alerts_again():
    base, cam = scene(), Cam(relearn_s=10)
    cam.show(base, seconds=6)
    vandal_visits(cam, base)
    bad = graffiti(base)
    cam.show(bad, seconds=30)                                     # долго испорчена — одно событие, потом это эталон
    assert len(cam.events) == 1
    vandal_visits(cam, bad, pid=9)
    worse = bad.copy()
    worse[DY1:DY2, DX1:DX2] = 255 - worse[DY1:DY2, DX1:DX2]
    cam.show(worse, seconds=10)
    assert len(cam.events) == 2 and cam.events[1].participants[0].track_id == 9


def test_whole_scene_shift_is_not_object_change():
    """Камеру сдвинули/панорама: дверь «уехала» из зоны, но изменилась вся сцена — это не порча объекта."""
    base, cam = scene(), Cam()
    cam.show(base, seconds=6)
    vandal_visits(cam, base)
    cam.show(np.roll(base, (25, 60), (0, 1)), seconds=12)
    assert cam.events == []
