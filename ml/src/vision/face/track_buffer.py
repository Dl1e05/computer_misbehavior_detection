"""TrackBuffer: короткая память кропов верхней части людей — только в ОЗУ.

Зачем: лицо нужно лишь после подтверждённого события, но к этому моменту человек мог
уже отвернуться. Поэтому держим кропы последних ~5 с каждого трека, а лица в них
НЕ ищем — это делает FaceService и только по команде ParticipantCollector.

- каждый every_n-й кадр трека -> копия верхней части рамки (upper_frac высоты);
- кропы старше keep_s выбрасываются;
- трек не виден дольше lost_s -> все его кропы удаляются;
- на диск ничего не пишется.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime

import numpy as np


@dataclass
class Crop:
    ts: datetime
    image: np.ndarray            # BGR, своя копия (кадр потом переиспользуется)


def upper_crop(image: np.ndarray, bbox, upper_frac: float = 0.5, pad: float = 0.1) -> np.ndarray | None:
    """Верх рамки человека (там голова) с небольшим запасом по бокам."""
    h, w = image.shape[:2]
    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1
    if bw <= 1 or bh <= 1:
        return None
    cx1 = int(max(0, x1 - pad * bw))
    cx2 = int(min(w, x2 + pad * bw))
    cy1 = int(max(0, y1 - pad * bh * upper_frac))
    cy2 = int(min(h, y1 + upper_frac * bh))
    if cx2 - cx1 < 2 or cy2 - cy1 < 2:
        return None
    return image[cy1:cy2, cx1:cx2].copy()


class TrackBuffer:
    def __init__(self, keep_s: float = 5.0, every_n: int = 3, upper_frac: float = 0.5,
                 lost_s: float = 2.0, max_per_track: int = 64):
        self.keep_s = keep_s
        self.every_n = max(1, int(every_n))
        self.upper_frac = upper_frac
        self.lost_s = lost_s
        self._crops: dict[int, deque[Crop]] = {}
        self._frames: dict[int, int] = {}          # сколько кадров трек уже виден (для every_n)
        self._last_seen: dict[int, datetime] = {}
        self.max_per_track = max_per_track

    def update(self, ts: datetime, image: np.ndarray, tracks) -> None:
        for t in tracks:
            tid = t.track_id
            n = self._frames.get(tid, 0)
            self._frames[tid] = n + 1
            self._last_seen[tid] = ts
            if n % self.every_n:
                continue
            crop = upper_crop(image, t.bbox, self.upper_frac)
            if crop is not None:
                self._crops.setdefault(tid, deque(maxlen=self.max_per_track)).append(Crop(ts, crop))
        self._prune(ts)

    def _prune(self, now: datetime) -> None:
        for tid in list(self._last_seen):
            if (now - self._last_seen[tid]).total_seconds() > self.lost_s:
                self.drop(tid)
        for q in self._crops.values():
            while q and (now - q[0].ts).total_seconds() > self.keep_s:
                q.popleft()

    def take(self, track_id: int) -> list[Crop]:
        """Кропы трека за последние keep_s (копия списка; буфер не меняется)."""
        return list(self._crops.get(track_id, ()))

    def drop(self, track_id: int) -> None:
        self._crops.pop(track_id, None)
        self._frames.pop(track_id, None)
        self._last_seen.pop(track_id, None)

    def reset(self) -> None:
        """Переподключение камеры: старые track_id недействительны."""
        self._crops.clear()
        self._frames.clear()
        self._last_seen.clear()

    def __len__(self) -> int:
        return sum(len(q) for q in self._crops.values())
