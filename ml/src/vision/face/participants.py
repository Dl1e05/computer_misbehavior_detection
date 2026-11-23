"""ParticipantCollector — ЕДИНСТВЕННОЕ место, откуда в конвейере запускается FaceService.

Ограничение ТЗ (5.1 п.5, 9.1, 13.1): лица — только у Участников уже подтверждённого
события. Поэтому:
- в каждом кадре collector лишь кладёт кропы в TrackBuffer (лица не ищутся);
- on_event() принимает событие и сам проверяет, что оно подтверждающее (FACE_EVENTS);
  suspicion и события без участников игнорируются;
- для каждого участника собираем кропы из буфера (~5 с до) + следующие after_s секунд,
  затем ОДИН вызов FaceService.best_face -> сообщение участника.

Сообщение участника (контракт шага 8): event_id, track_id, snapshot_ref,
snapshot_quality, embedding, embedding_model_version.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np

from vision.face.face_service import NO_USABLE_FACE, FaceService
from vision.face.storage import SnapshotStorage, snapshot_key
from vision.face.track_buffer import Crop, TrackBuffer, upper_crop

# (incident_type, event_type), после которых считаем лица. Это код, а не настройка — намеренно.
FACE_EVENTS = frozenset({
    ("zone_intrusion", "detection"),
    ("fall", "detection"),
    ("fight", "confirmation"),
})


def is_face_event(incident_type: str, event_type: str) -> bool:
    return (incident_type, event_type) in FACE_EVENTS


@dataclass
class ParticipantMessage:
    event_id: str
    camera_id: str
    track_id: int
    snapshot_ref: str | None
    snapshot_quality: float | str           # 0..1 или "no_usable_face"
    embedding: list[float] | None
    embedding_model_version: str
    reason: str | None = None               # для лога/отладки, в контракт не входит

    def to_dict(self) -> dict:
        return {"event_id": self.event_id, "track_id": self.track_id,
                "snapshot_ref": self.snapshot_ref, "snapshot_quality": self.snapshot_quality,
                "embedding": self.embedding, "embedding_model_version": self.embedding_model_version}


@dataclass
class _Job:
    event_id: str
    track_id: int
    deadline: datetime
    crops: list[Crop] = field(default_factory=list)
    seen_ids: set[int] = field(default_factory=set)


class ParticipantCollector:
    """Один на камеру (как трекер): буфер кропов + ожидающие задачи по участникам."""

    def __init__(self, camera_id: str, service: FaceService, storage: SnapshotStorage,
                 buffer: TrackBuffer | None = None, after_s: float = 2.0):
        self.camera_id = camera_id
        self.service = service
        self.storage = storage
        self.buffer = buffer or TrackBuffer()
        self.after_s = after_s
        self._jobs: list[_Job] = []

    @property
    def pending(self) -> int:
        return len(self._jobs)

    def on_frame(self, ts: datetime, image: np.ndarray, tracks) -> list[ParticipantMessage]:
        """Каждый кадр: пополнить буфер; дособрать кропы для ожидающих; закрыть созревшие задачи."""
        self.buffer.update(ts, image, tracks)
        if self._jobs:
            by_id = {t.track_id: t for t in tracks}
            for job in self._jobs:
                t = by_id.get(job.track_id)
                if t is not None:
                    crop = upper_crop(image, t.bbox, self.buffer.upper_frac)
                    if crop is not None:
                        job.crops.append(Crop(ts, crop))
        return self._finish(ts)

    def on_event(self, detection, event_id: str) -> int:
        """Событие от правила. Возвращает, сколько участников поставлено в очередь на лицо."""
        if not is_face_event(detection.incident_type, detection.event_type):
            return 0
        n = 0
        deadline = detection.occurred_at + timedelta(seconds=self.after_s)
        for p in detection.participants:
            if any(j.track_id == p.track_id and j.event_id == event_id for j in self._jobs):
                continue
            self._jobs.append(_Job(event_id, p.track_id, deadline, self.buffer.take(p.track_id)))
            n += 1
        return n

    def flush(self) -> list[ParticipantMessage]:
        """Конец работы: закрыть все задачи с тем, что успели собрать."""
        return self._finish(None)

    def reset(self) -> list[ParticipantMessage]:
        """Переподключение камеры: track_id недействительны — закрываем задачи, чистим буфер."""
        out = self.flush()
        self.buffer.reset()
        return out

    def _finish(self, now: datetime | None) -> list[ParticipantMessage]:
        ready = [j for j in self._jobs if now is None or now >= j.deadline]
        if not ready:
            return []
        self._jobs = [j for j in self._jobs if j not in ready]
        return [self._process(j) for j in ready]

    def _process(self, job: _Job) -> ParticipantMessage:
        res = self.service.best_face([c.image for c in job.crops])
        version = self.service.model_version
        if not res.usable:
            return ParticipantMessage(job.event_id, self.camera_id, job.track_id, None,
                                      NO_USABLE_FACE, None, version, res.reason)
        ts = job.crops[-1].ts if job.crops else job.deadline
        ref = self.storage.put(snapshot_key(self.camera_id, job.event_id, job.track_id, ts), res.jpeg)
        return ParticipantMessage(job.event_id, self.camera_id, job.track_id, ref, res.quality,
                                  [round(float(v), 6) for v in res.embedding], version)
