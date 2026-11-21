"""Поза: yolo26n-pose батчем, ТОЛЬКО для камер, где правилу нужны poses (падение, драка).

План А (по ТЗ): люди и треки — от детектора yolo26n, скелеты — от pose-модели на тех же
кадрах; скелет привязывается к треку по IoU рамок (attach_poses).
План Б (pose-модель вместо детектора на этих камерах) — позже, замер в шаге 9.

17 точек COCO, каждая (x, y, conf) в пикселях кадра. Номера точек — константы ниже.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

# COCO-17
NOSE = 0
L_SHOULDER, R_SHOULDER = 5, 6
L_ELBOW, R_ELBOW = 7, 8
L_WRIST, R_WRIST = 9, 10
L_HIP, R_HIP = 11, 12
L_KNEE, R_KNEE = 13, 14
L_ANKLE, R_ANKLE = 15, 16


@dataclass
class Pose:
    bbox: tuple[float, float, float, float]   # рамка pose-модели, пиксели
    conf: float
    keypoints: np.ndarray                     # (17, 3): x, y, conf


class PoseEstimator:
    def __init__(self, weights: str = "weights/yolo26n-pose.pt", imgsz: int = 640,
                 conf: float = 0.25, device: int | str = 0, half: bool = True):
        from ultralytics import YOLO

        self.model = YOLO(weights)
        fp16 = half and device != "cpu"
        self.kw = dict(imgsz=imgsz, conf=conf, device=device,
                       quantize=16 if fp16 else None, verbose=False)

    @classmethod
    def from_config(cls, cfg: dict) -> "PoseEstimator":
        return cls(**cfg)

    def warmup(self, batch: int = 1) -> None:
        self([np.zeros((480, 640, 3), dtype=np.uint8)] * batch)

    def __call__(self, images: Sequence[np.ndarray]) -> list[list[Pose]]:
        if not images:
            return []
        out = []
        for r in self.model.predict(list(images), **self.kw):
            if r.boxes is None or len(r.boxes) == 0 or r.keypoints is None:
                out.append([])
                continue
            boxes = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            kpts = r.keypoints.data.cpu().numpy()          # (N, 17, 3)
            out.append([Pose(tuple(map(float, b)), float(c), k) for b, c, k in zip(boxes, confs, kpts)])
        return out


def iou(a, b) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0:
        return 0.0
    area = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / area if area > 0 else 0.0


def attach_poses(tracks, poses: list[Pose], min_iou: float = 0.5) -> dict[int, np.ndarray]:
    """Скелет -> трек: жадно по убыванию IoU рамок, каждый скелет и трек — не больше одного раза."""
    pairs = sorted(((iou(t.bbox, p.bbox), ti, pi)
                    for ti, t in enumerate(tracks) for pi, p in enumerate(poses)), reverse=True)
    used_t, used_p, out = set(), set(), {}
    for score, ti, pi in pairs:
        if score < min_iou:
            break
        if ti in used_t or pi in used_p:
            continue
        used_t.add(ti)
        used_p.add(pi)
        out[tracks[ti].track_id] = poses[pi].keypoints
    return out
