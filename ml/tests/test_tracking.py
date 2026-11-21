from pathlib import Path

import numpy as np
import pytest

from vision.analytics.tracker import CameraTracker, seconds_to_frames
from vision.inference.detector import boxes_to_array

IMG = np.zeros((480, 640, 3), dtype=np.uint8)
WEIGHTS = Path(__file__).resolve().parents[1] / "weights" / "yolo26n.pt"


def person(x, y, conf=0.9):
    return [x, y, x + 40, y + 100, conf, 0]


def run(tracker, frames, session=1):
    """frames: список кадров, кадр = список рамок. Возвращает id-шники по кадрам."""
    out = []
    for boxes in frames:
        dets = np.array(boxes, dtype=np.float32).reshape(-1, 6)
        out.append([t.track_id for t in tracker.update(dets, IMG, session=session)])
    return out


def test_seconds_to_frames():
    assert seconds_to_frames(2.0, 8) == 16
    assert seconds_to_frames(0.01, 8) == 1


def test_moving_person_keeps_one_id():
    tr = CameraTracker("cam-01", fps=8)
    ids = run(tr, [[person(100 + 3 * i, 100)] for i in range(20)])
    ids = [i for frame in ids for i in frame]
    assert ids and len(set(ids)) == 1


def test_two_people_two_ids():
    tr = CameraTracker("cam-01", fps=8)
    ids = run(tr, [[person(100 + 2 * i, 100), person(400 - 2 * i, 200)] for i in range(10)])
    assert len(ids[-1]) == 2 and len(set(ids[-1])) == 2


def test_cameras_do_not_share_tracker():
    a, b = CameraTracker("cam-01", fps=8), CameraTracker("cam-02", fps=8)
    run(a, [[person(100 + 2 * i, 100)] for i in range(10)])
    # во второй камере человек в ДРУГОМ месте — трек камеры A его не «перехватит»
    ids_b = run(b, [[person(400 + 2 * i, 300)] for i in range(10)])
    assert ids_b[-1] and len(a._tracker.active_tracks) == 1 and len(b._tracker.active_tracks) == 1


def test_short_occlusion_keeps_id():
    tr = CameraTracker("cam-01", fps=8, lost_s=2.0)          # помним 16 кадров
    before = run(tr, [[person(100 + 2 * i, 100)] for i in range(10)])
    run(tr, [[] for _ in range(4)])                           # пропал на 0.5 с
    after = run(tr, [[person(128 + 2 * i, 100)] for i in range(6)])
    assert after[-1] == before[-1]


def test_new_session_resets_ids():
    tr = CameraTracker("cam-01", fps=8)
    first = run(tr, [[person(100 + 2 * i, 100)] for i in range(10)], session=1)
    second = run(tr, [[person(100 + 2 * i, 100)] for i in range(10)], session=2)
    assert first[-1] and second[-1]
    assert len(tr._tracker.active_tracks) == 1               # старые треки сброшены


def test_empty_boxes_to_array():
    assert boxes_to_array(None).shape == (0, 6)


@pytest.mark.skipif(not WEIGHTS.exists(), reason="нет weights/yolo26n.pt")
def test_batch_detector_one_result_per_frame():
    from vision.inference.detector import BatchDetector
    import torch

    det = BatchDetector(str(WEIGHTS), device=0 if torch.cuda.is_available() else "cpu")
    out = det([IMG, IMG, np.zeros((240, 320, 3), np.uint8)])
    assert len(out) == 3 and all(o.shape[1] == 6 for o in out)


def test_bus_without_drop_keeps_every_frame(tmp_path):
    import cv2
    from vision.runtime.bus import FrameBus
    from vision.sources.video_source import VideoSource

    path = tmp_path / "v.avi"
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 25, (64, 48))
    for _ in range(50):
        w.write(np.zeros((48, 64, 3), np.uint8))
    w.release()

    srcs = [VideoSource(f"cam-{i}", str(path), process_fps=5, realtime=False) for i in range(3)]
    bus = FrameBus(srcs, drop_old=False).start()
    got = {}
    try:
        while not bus.done:
            for f in bus.next_batch(timeout=2):
                got[f.camera_id] = got.get(f.camera_id, 0) + 1
    finally:
        bus.stop()
    assert got == {"cam-0": 10, "cam-1": 10, "cam-2": 10}     # 2 с * 5 кадр/с на каждую камеру
