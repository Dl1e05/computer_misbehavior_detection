"""Шаг 9: счётчики пропусков в шине, задержка кадра и расчёт узлов на 100 камер (без GPU)."""
from __future__ import annotations

import importlib.util
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from vision.runtime.bus import FrameBus
from vision.sources.video_source import Frame

ML = Path(__file__).resolve().parents[1]


def _load_test_module():
    spec = importlib.util.spec_from_file_location("load_test", ML / "scripts" / "load_test.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class FakeSource:
    def __init__(self, cid, n, period):
        self.camera_id, self.n, self.period = cid, n, period

    def __iter__(self):
        for i in range(self.n):
            time.sleep(self.period)
            yield Frame(self.camera_id, datetime.now(timezone.utc), np.zeros((2, 2, 3), np.uint8), i, 1)

    def stop(self):
        pass


def test_frame_has_read_time():
    t = time.monotonic()
    f = Frame("c", datetime.now(timezone.utc), np.zeros((1, 1, 3), np.uint8), 0, 1)
    assert t <= f.t_read <= time.monotonic()


def test_bus_counts_dropped_frames_when_consumer_is_slow():
    bus = FrameBus([FakeSource("a", 40, 0.005)], drop_old=True).start()
    got = 0
    while not bus.done:
        got += len(bus.next_batch(timeout=0.5))
        time.sleep(0.05)                                   # «модель» медленнее камеры
    bus.stop()
    assert bus.produced["a"] == 40
    assert bus.dropped["a"] > 0 and got + bus.dropped["a"] == 40


def test_bus_no_drops_when_fast_enough():
    bus = FrameBus([FakeSource("a", 20, 0.01)], drop_old=True).start()
    while not bus.done:
        bus.next_batch(timeout=0.5)
    bus.stop()
    assert bus.dropped["a"] == 0


def test_report_counts_nodes_for_100_cameras():
    lt = _load_test_module()
    env = {"gpu": "GPU", "driver": "1", "cpu": "cpu", "power_plugged": "сеть", "torch": "t", "ultralytics": "u"}
    rows = [{"mode": "persons", "cams": n, "ok": ok, "why_not": "" if ok else "fps"} for n, ok in
            [(4, True), (8, True), (12, False)]] + \
           [{"mode": "pose", "cams": 4, "ok": False, "why_not": "fps"}]
    md = lt.report_md(rows, env, Path("x.csv"), 8)
    assert "предел ≥ 8 камер (сломалось на 12)" in md and "**18** таких узлов" in md   # ceil(100 / 5.6)
    assert "**pose**: не тянет даже 4" in md
