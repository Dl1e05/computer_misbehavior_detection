"""Огонь и дым: отдельная YOLO26, веса — после обучения на D-Fire (Фаза 3.1).

Пока весов нет, FireDetector.available == False: правило огня выключается, и это
видно в статусе узла — остальной модуль работает дальше (не падает).

На выходе для каждого кадра — (N, 6): x1, y1, x2, y2, conf, cls в пикселях;
cls — индекс в self.names (у D-Fire: 0 smoke, 1 fire — проверено по разметке).
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from vision.inference.detector import boxes_to_array


class FireDetector:
    def __init__(self, weights: str = "weights/fire_yolo26n.pt", imgsz: int = 640,
                 conf: float = 0.25, device: int | str = 0, half: bool = True):
        self.weights = weights
        self.model = None
        self.names: dict[int, str] = {}
        self.reason = ""
        if not Path(weights).exists():
            self.reason = f"нет весов {weights} (обучение — Фаза 3.1)"
            return
        from ultralytics import YOLO

        self.model = YOLO(weights)
        self.names = dict(self.model.names)
        fp16 = half and device != "cpu"
        self.kw = dict(imgsz=imgsz, conf=conf, device=device,
                       quantize=16 if fp16 else None, verbose=False)

    @property
    def available(self) -> bool:
        return self.model is not None

    @classmethod
    def from_config(cls, cfg: dict) -> "FireDetector":
        return cls(**cfg)

    def warmup(self, batch: int = 1) -> None:
        if self.available:
            self([np.zeros((480, 640, 3), dtype=np.uint8)] * batch)

    def __call__(self, images: Sequence[np.ndarray]) -> list[np.ndarray]:
        if not self.available or not images:
            return [np.empty((0, 6), np.float32) for _ in images]
        return [boxes_to_array(r.boxes) for r in self.model.predict(list(images), **self.kw)]


class MockFireDetector:
    """Для тестов и демо: выдаёт заранее заданную последовательность рамок.

    script: список «прогонов», каждый — список рамок (x1, y1, x2, y2, conf, cls) в пикселях.
    Когда список кончился — дальше пусто.
    """

    def __init__(self, script: list[list[tuple]], names: dict[int, str] | None = None):
        self.script = list(script)
        self.names = names or {0: "smoke", 1: "fire"}
        self.reason = ""
        self.available = True

    def warmup(self, batch: int = 1) -> None:
        pass

    def __call__(self, images: Sequence[np.ndarray]) -> list[np.ndarray]:
        out = []
        for _ in images:
            boxes = self.script.pop(0) if self.script else []
            out.append(np.array(boxes, dtype=np.float32).reshape(-1, 6))
        return out
