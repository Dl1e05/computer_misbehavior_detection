"""Шаг 7: буфер кропов, качество лица, FaceService и приватность (лица только после подтверждения).

Почти всё — на заглушке InsightFace (без GPU и без чужих лиц). Один тест в конце гоняет
настоящую buffalo_l на образцах, которые идут в пакете insightface; пропускается, если модели нет.
"""
from __future__ import annotations

import pathlib
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from vision.analytics.tracker import Track
from vision.face.face_service import NO_USABLE_FACE, FaceService
from vision.face.participants import FACE_EVENTS, ParticipantCollector, is_face_event
from vision.face.quality import QualityParams, face_quality, frontality
from vision.face.storage import LocalSnapshotStorage, snapshot_key
from vision.face.track_buffer import TrackBuffer, upper_crop
from vision.rules.base import Detection, Participant

T0 = datetime(2026, 11, 24, 12, 0, tzinfo=timezone.utc)
FPS = 8
SRC = pathlib.Path(__file__).resolve().parents[1] / "src"


def textured(h=480, w=640, seed=0):
    return np.random.default_rng(seed).integers(0, 255, (h, w, 3), dtype=np.uint8)


@dataclass
class FakeFace:
    bbox: tuple
    det_score: float = 0.9
    kps: tuple = ((30, 40), (70, 40), (50, 60), (35, 80), (65, 80))   # анфас
    normed_embedding: np.ndarray | None = None


class FakeApp:
    """Заглушка FaceAnalysis: лицо 80×100 в левом верхнем углу кропа, если кроп достаточно большой."""

    def __init__(self, score=0.9, min_h=100):
        self.score, self.min_h, self.calls = score, min_h, 0

    def get(self, img):
        self.calls += 1
        if img.shape[0] < self.min_h:
            return []
        v = np.random.default_rng(int(img.sum()) % 1000).normal(size=512).astype(np.float32)
        return [FakeFace((10, 0, 90, 100), self.score, normed_embedding=v / np.linalg.norm(v))]


def tracks(*items):
    return [Track(tid, tuple(map(float, bbox)), 0.9, 0) for tid, bbox in items]


def detection(incident, event, tids, ts=T0):
    return Detection(incident, event, "cam-01", ts, 0.8,
                     [Participant(t, (0.1, 0.1, 0.3, 0.9), 0.9) for t in tids])


def make_collector(tmp_path, app=None, after_s=2.0):
    service = FaceService(app or FakeApp())
    storage = LocalSnapshotStorage(tmp_path / "snapshots")
    return ParticipantCollector("cam-01", service, storage, TrackBuffer(keep_s=5, every_n=3), after_s), service


def stream(col, n_frames, start=0, people=((1, (100, 50, 220, 450)), (2, (400, 60, 520, 460))), on_frame_out=None):
    out = []
    img = textured()
    for i in range(start, start + n_frames):
        out += col.on_frame(T0 + timedelta(seconds=i / FPS), img, tracks(*people))
    return out


# ---------- TrackBuffer ----------

def test_upper_crop_is_top_of_bbox_and_a_copy():
    img = textured()
    c = upper_crop(img, (100, 100, 200, 300), upper_frac=0.5, pad=0.0)
    assert c.shape == (100, 100, 3)
    c[:] = 0
    assert img[100:200, 100:200].any()            # кадр не испорчен — у буфера своя копия


def test_buffer_every_n_keep_s_and_drop_lost():
    b = TrackBuffer(keep_s=5, every_n=3, lost_s=2)
    img = textured()
    for i in range(80):                            # 10 с при 8 кадр/с
        b.update(T0 + timedelta(seconds=i / FPS), img, tracks((1, (10, 10, 110, 310))))
    crops = b.take(1)
    assert 12 <= len(crops) <= 15                  # каждый 3-й кадр за последние ~5 с
    assert (crops[-1].ts - crops[0].ts).total_seconds() <= 5
    t_end = T0 + timedelta(seconds=80 / FPS)
    b.update(t_end + timedelta(seconds=2.5), img, [])   # трек пропал дольше lost_s
    assert b.take(1) == [] and len(b) == 0


# ---------- качество ----------

def test_quality_rejects_small_and_unsure():
    img = textured()
    qp = QualityParams()
    assert face_quality(FakeFace((0, 0, 40, 40)), img, qp)[1] == "too_small"
    assert face_quality(FakeFace((0, 0, 100, 100), det_score=0.4), img, qp)[1] == "low_det_score"
    q, why = face_quality(FakeFace((0, 0, 100, 100)), img, qp)
    assert why is None and 0 < q <= 1


def test_frontal_face_scores_higher_than_profile():
    front = ((30, 40), (70, 40), (50, 60), (35, 80), (65, 80))
    profile = ((30, 40), (50, 40), (58, 60), (35, 80), (55, 80))
    assert frontality(front) == pytest.approx(1.0)
    assert frontality(profile) < 0.3


# ---------- FaceService ----------

def test_best_face_gives_unit_512_vector_and_jpeg():
    s = FaceService(FakeApp())
    r = s.best_face([textured(200, 150, i) for i in range(5)])
    assert r.usable and r.embedding.shape == (512,)
    assert np.linalg.norm(r.embedding) == pytest.approx(1.0, abs=1e-5)
    assert r.jpeg[:2] == b"\xff\xd8"               # JPEG
    assert s.calls == 1


def test_no_face_gives_no_usable_face():
    s = FaceService(FakeApp(min_h=10_000))
    r = s.best_face([textured(200, 150)])
    assert r.quality == NO_USABLE_FACE and r.embedding is None and r.jpeg is None and r.reason == "no_face"
    assert FaceService(FakeApp()).best_face([]).reason == "no_crops"


def test_max_crops_limits_work():
    app = FakeApp()
    FaceService(app, max_crops=4).best_face([textured(200, 150, i) for i in range(30)])
    assert app.calls == 4


# ---------- приватность: лица только после подтверждения ----------

def test_privacy_plain_stream_no_face_calls(tmp_path):
    col, service = make_collector(tmp_path)
    assert stream(col, 200) == []
    assert service.calls == 0 and service.app.calls == 0
    assert not (tmp_path / "snapshots").exists()   # на диск ничего


@pytest.mark.parametrize("incident,event", [("fight", "suspicion"), ("fire", "detection"),
                                            ("unknown", "detection")])
def test_privacy_not_confirming_events_ignored(tmp_path, incident, event):
    col, service = make_collector(tmp_path)
    stream(col, 40)
    assert col.on_event(detection(incident, event, [1, 2], T0 + timedelta(seconds=5)), "ev-1") == 0
    stream(col, 40, start=40)
    assert col.flush() == []
    assert service.calls == 0 and service.app.calls == 0


def test_face_events_set_is_exactly_the_confirming_ones():
    assert FACE_EVENTS == {("zone_intrusion", "detection"), ("fall", "detection"), ("fight", "confirmation")}
    assert not is_face_event("fight", "suspicion")


@pytest.mark.parametrize("incident,event", sorted(FACE_EVENTS))
def test_confirmed_event_gives_participant_messages_after_delay(tmp_path, incident, event):
    col, service = make_collector(tmp_path, after_s=2.0)
    stream(col, 40)                                            # 5 с до события
    ts = T0 + timedelta(seconds=40 / FPS)
    assert col.on_event(detection(incident, event, [1], ts), "ev-42") == 1
    assert service.calls == 0                                  # лицо не тормозит тревогу
    msgs = stream(col, 8, start=40)                            # 1 с — ещё рано
    assert msgs == [] and col.pending == 1
    msgs = stream(col, 10, start=48)                           # прошло 2 с
    assert len(msgs) == 1 and col.pending == 0
    m = msgs[0]
    assert (m.event_id, m.track_id) == ("ev-42", 1)            # только участник события, не трек 2
    assert service.calls == 1
    assert len(m.embedding) == 512 and np.linalg.norm(m.embedding) == pytest.approx(1.0, abs=1e-4)
    assert m.embedding_model_version == "insightface-buffalo_l-1.0"
    assert 0 < m.snapshot_quality <= 1
    assert m.snapshot_ref == snapshot_key("cam-01", "ev-42", 1, ts) == "faces/cam-01/20261124/ev-42_1.jpg"
    assert (tmp_path / "snapshots" / m.snapshot_ref).read_bytes()[:2] == b"\xff\xd8"
    assert set(m.to_dict()) == {"event_id", "track_id", "snapshot_ref", "snapshot_quality",
                                "embedding", "embedding_model_version"}


def test_confirmed_event_without_face_sends_no_usable_face(tmp_path):
    col, service = make_collector(tmp_path, app=FakeApp(min_h=10_000))
    stream(col, 20)
    col.on_event(detection("fall", "detection", [2], T0 + timedelta(seconds=2.5)), "ev-7")
    msgs = stream(col, 30, start=20)
    assert len(msgs) == 1
    m = msgs[0]
    assert m.snapshot_quality == NO_USABLE_FACE and m.snapshot_ref is None and m.embedding is None
    assert not (tmp_path / "snapshots").exists()


def test_participant_seen_only_after_event_still_collected(tmp_path):
    """Человек появился в кадре только в момент события: кропы берутся из следующих 2 с."""
    col, service = make_collector(tmp_path)
    col.on_event(detection("zone_intrusion", "detection", [5], T0), "ev-9")
    msgs = stream(col, 20, people=((5, (100, 50, 220, 450)),))
    assert len(msgs) == 1 and msgs[0].embedding is not None


def test_face_service_called_only_from_collector_and_enroll():
    """«Видно в коде»: best_face вызывается только в participants.py (enroll — свой метод)."""
    callers = [p.relative_to(SRC).as_posix() for p in SRC.rglob("*.py")
               if ".best_face(" in p.read_text(encoding="utf-8")]
    assert callers == ["vision/face/participants.py"]
    scripts = SRC.parent / "scripts"
    enroll_users = sorted(p.name for p in scripts.glob("*.py") if ".enroll(" in p.read_text(encoding="utf-8"))
    assert enroll_users == ["calibrate_faces.py", "enroll_faces.py"]   # только фото, не поток камеры
    assert not any(".enroll(" in p.read_text(encoding="utf-8") for p in SRC.rglob("*.py")
                   if p.name != "face_service.py")


def test_storage_is_write_only_and_stays_inside_root(tmp_path):
    st = LocalSnapshotStorage(tmp_path)
    assert st.put("faces/a.jpg", b"x") == "faces/a.jpg"
    assert not any(hasattr(st, m) for m in ("get", "read", "list", "delete"))
    with pytest.raises(ValueError):
        st.put("../escape.jpg", b"x")


# ---------- настоящая InsightFace (образцы из пакета insightface) ----------

def _buffalo_available() -> bool:
    return (pathlib.Path.home() / ".insightface" / "models" / "buffalo_l" / "w600k_r50.onnx").exists()


@pytest.mark.skipif(not _buffalo_available(), reason="нет модели buffalo_l")
def test_real_insightface_vector_shape_and_enroll_rules():
    import cv2
    from insightface.data import get_image

    service = FaceService.from_config({"det_size": 640})
    single = get_image("Tom_Hanks_54745")                     # образец из пакета insightface, 112×112
    single = cv2.copyMakeBorder(single, 100, 100, 100, 100, cv2.BORDER_CONSTANT, value=(127, 127, 127))
    r = service.enroll(single)
    assert r.usable, r.reason
    assert r.embedding.shape == (512,) and np.linalg.norm(r.embedding) == pytest.approx(1.0, abs=1e-3)
    same = service.enroll(cv2.flip(single, 1))                # тот же человек, зеркально
    assert float(r.embedding @ same.embedding) > 0.5
    group = service.enroll(get_image("t1"))                   # групповое фото — для базы не годится
    assert group.quality == NO_USABLE_FACE and group.reason == "several_faces"
    assert service.enroll(get_image("t1"), pick="center").usable          # калибровка: лицо у центра
    assert service.enroll(np.zeros((480, 640, 3), np.uint8)).reason == "no_face"
