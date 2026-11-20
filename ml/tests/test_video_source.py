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