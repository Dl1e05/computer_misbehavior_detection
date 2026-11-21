import time
from datetime import timedelta

import cv2
import numpy as np
import pytest

from vision.sources.video_source import OFFLINE, ONLINE, VideoSource, is_live, mask_password


@pytest.fixture
def video_file(tmp_path):
    """Синтетический ролик: 2 секунды, 25 кадров/с, 64x48."""
    path = tmp_path / "demo.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 25, (64, 48))
    for i in range(50):
        writer.write(np.full((48, 64, 3), i * 5 % 255, dtype=np.uint8))
    writer.release()
    return path


def test_file_respects_process_fps(video_file):
    statuses = []
    src = VideoSource("cam-01", str(video_file), process_fps=5, realtime=False,
                      on_status=lambda cam, s: statuses.append(s))
    frames = list(src)

    assert len(frames) == 10                         # 2 с * 5 кадр/с
    assert all(f.camera_id == "cam-01" for f in frames)
    gaps = [b.ts - a.ts for a, b in zip(frames, frames[1:])]
    assert all(g == timedelta(milliseconds=200) for g in gaps)
    assert frames[0].ts.tzinfo is not None           # время с зоной
    assert statuses == [ONLINE, OFFLINE]


def test_ts_iso_has_millis_and_zone(video_file):
    f = next(iter(VideoSource("cam-01", str(video_file), realtime=False)))
    assert f.ts_iso.endswith("+00:00")
    assert "." in f.ts_iso and len(f.ts_iso.split(".")[1]) == 3 + 6   # .123+00:00


def test_missing_file_goes_offline(tmp_path):
    statuses = []
    src = VideoSource("cam-01", str(tmp_path / "nope.mp4"),
                      on_status=lambda cam, s: statuses.append(s))
    src.status = ONLINE   # чтобы увидеть переход
    assert list(src) == []
    assert statuses == [OFFLINE]


def test_env_vars_in_source(monkeypatch):
    monkeypatch.setenv("CAM01_PASS", "secret")
    src = VideoSource("cam-01", "rtsp://admin:${CAM01_PASS}@10.0.0.11/stream2")
    assert src.source == "rtsp://admin:secret@10.0.0.11/stream2"
    assert src.live
    assert mask_password(src.source) == "rtsp://admin:***@10.0.0.11/stream2"


def test_is_live():
    assert is_live("rtsp://10.0.0.1/s")
    assert is_live("0")
    assert not is_live("data/demo.mp4")


class FakeCap:
    """Имитация RTSP: отдаёт n кадров, потом read() -> False (обрыв)."""

    def __init__(self, n):
        self.n = n

    def read(self):
        if self.n <= 0:
            return False, None
        self.n -= 1
        time.sleep(0.005)                 # камера отдаёт кадры не мгновенно
        return True, np.zeros((4, 4, 3), dtype=np.uint8)

    def release(self):
        pass


def test_live_reconnect_offline_online(monkeypatch):
    # 1-е подключение: 3 кадра и обрыв; 2-е: камера недоступна; 3-е: снова кадры
    caps = iter([FakeCap(3), None, FakeCap(10_000)])
    monkeypatch.setattr(VideoSource, "_open", lambda self: next(caps, None))

    statuses = []
    src = VideoSource("cam-01", "rtsp://localhost:8554/cam1", process_fps=1000,
                      fail_threshold=2, min_backoff_s=0.01,
                      on_status=lambda cam, s: statuses.append(s))
    frames = []
    deadline = time.monotonic() + 10
    for f in src:
        frames.append(f)
        if (f.session == 2 and f.index >= 2) or time.monotonic() > deadline:
            break

    assert statuses[:3] == [ONLINE, OFFLINE, ONLINE]
    assert {f.session for f in frames} == {1, 2}
    second = [f for f in frames if f.session == 2]
    assert second[0].index == 0          # новая сессия -> счёт заново, трекер сбросить
    assert all(f.ts.tzinfo is not None for f in frames)
