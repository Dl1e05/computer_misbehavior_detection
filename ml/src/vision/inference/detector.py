"""Стадия Inference: одна YOLO на GPU обрабатывает кадры всех камер одним батчем.

На выходе для каждого кадра — массив (N, 6): x1, y1, x2, y2, conf, cls.
Порядок совпадает с порядком входных кадров: i-й результат -> i-я камера.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

PERSON = 0  # класс COCO


def boxes_to_array(boxes) -> np.ndarray:
    """ultralytics Boxes -> (N, 6) float32; пустой результат -> (0, 6)."""
    if boxes is None or len(boxes) == 0:
        return np.empty((0, 6), dtype=np.float32)
    return np.hstack([
        boxes.xyxy.cpu().numpy(),
        boxes.conf.cpu().numpy()[:, None],
        boxes.cls.cpu().numpy()[:, None],
    ]).astype(np.float32)


class BatchDetector:
    def __init__(self, weights: str = "weights/yolo26n.pt", imgsz: int = 640,
                 conf: float = 0.1, device: int | str = 0, half: bool = True):
        from ultralytics import YOLO   # тяжёлый импорт — только когда детектор реально нужен

        self.model = YOLO(weights)
        fp16 = half and device != "cpu"
        self.kw = dict(imgsz=imgsz, conf=conf, device=device,
                       quantize=16 if fp16 else None, verbose=False)

    def warmup(self, batch: int = 1) -> None:
        """Первый прогон инициализирует CUDA (секунды) — делаем его до старта камер."""
        self([np.zeros((480, 640, 3), dtype=np.uint8)] * batch)

    @classmethod
    def from_config(cls, cfg: dict) -> "BatchDetector":
        return cls(**cfg)

    def __call__(self, images: Sequence[np.ndarray],
                 classes: Sequence[int] = (PERSON,)) -> list[np.ndarray]:
        if not images:
            return []
        results = self.model.predict(list(images), classes=list(classes), **self.kw)
        return [boxes_to_array(r.boxes) for r in results]
