"""Стадия Analytics, часть 1: трекер людей ОДНОЙ камеры.

Правило: никогда не смешивать камеры в одном трекере — у каждой камеры свой
CameraTracker. track_id уникален только внутри камеры и текущей сессии.

Интерфейс CameraTracker.update(dets, image, session) -> list[Track] не зависит
от реализации: сейчас ByteTrack, план Б — BoT-SORT (параметр tracker.type).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class Track:
    track_id: int
    bbox: tuple[float, float, float, float]   # x1, y1, x2, y2 в пикселях кадра
    conf: float
    cls: int


def seconds_to_frames(seconds: float, fps: float) -> int:
    """Пороги храним в секундах, трекеру нужны кадры: смена FPS ничего не ломает."""
    return max(1, math.ceil(seconds * fps))


class CameraTracker:
    def __init__(self, camera_id: str, fps: float, type: str = "bytetrack",
                 track_thresh: float = 0.5, min_conf: float = 0.1,
                 min_hits: int = 3, lost_s: float = 2.0):
        if type != "bytetrack":
            raise ValueError(f"трекер '{type}' пока не поддержан (есть: bytetrack)")
        self.camera_id = camera_id
        self.fps = fps
        self._cfg = dict(track_thresh=track_thresh, min_conf=min_conf, min_hits=min_hits,
                         lost_frames=seconds_to_frames(lost_s, fps))
        self._session: int | None = None
        self._tracker = self._make()

    @classmethod
    def from_config(cls, camera_id: str, fps: float, cfg: dict) -> "CameraTracker":
        return cls(camera_id, fps, **cfg)

    def _make(self):
        from boxmot import ByteTrack
        from boxmot.trackers.bytetrack.config import ByteTrackConfig

        c = self._cfg
        return ByteTrack(ByteTrackConfig(
            track_thresh=c["track_thresh"], min_conf=c["min_conf"], min_hits=c["min_hits"],
            max_lost_frames=c["lost_frames"], max_age=c["lost_frames"],
        ))

    def reset(self) -> None:
        """После переподключения камеры старые track_id недействительны."""
        self._tracker = self._make()

    def update(self, dets: np.ndarray, image: np.ndarray, session: int | None = None) -> list[Track]:
        if session is not None and session != self._session:
            if self._session is not None:
                self.reset()
            self._session = session
        if dets is None or len(dets) == 0:
            dets = np.empty((0, 6), dtype=np.float32)
        out = self._tracker.update(dets, image)    # (M, 8): x1,y1,x2,y2,id,conf,cls,det_ind
        return [Track(int(t[4]), tuple(float(v) for v in t[:4]), float(t[5]), int(t[6]))
                for t in out]
