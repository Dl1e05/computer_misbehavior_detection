"""Оценка пригодности снимка лица (стартовые пороги — настраивать по своим записям).

Жёсткие отсечки: det_score < min_det_score или меньшая сторона лица < min_side_px -> лицо
не годится. Остальное — взвешенная сумма 0..1:
  уверенность детектора, размер, резкость (дисперсия Лапласиана), фронтальность
  (нос примерно посередине между глазами).
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class QualityParams:
    min_det_score: float = 0.6
    min_side_px: int = 48
    good_side_px: int = 112          # с этого размера «размер» = 1.0 (вход ArcFace — 112×112)
    good_sharpness: float = 150.0    # дисперсия Лапласиана, при которой «резкость» = 1.0
    w_det: float = 0.4
    w_size: float = 0.2
    w_sharp: float = 0.2
    w_frontal: float = 0.2


def frontality(kps) -> float:
    """1.0 — нос ровно между глазами (анфас), 0 — сильно в профиль. kps: 5×2 (глаза, нос, рот)."""
    kps = np.asarray(kps, dtype=np.float32)
    le, re, nose = kps[0], kps[1], kps[2]
    eye_dist = abs(re[0] - le[0])
    if eye_dist < 1e-3:
        return 0.0
    offset = abs(nose[0] - (le[0] + re[0]) / 2) / eye_dist     # 0 анфас, ~0.5 профиль
    return float(max(0.0, 1.0 - 2.0 * offset))


def sharpness(gray: np.ndarray) -> float:
    if gray.size == 0:
        return 0.0
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def face_quality(face, image: np.ndarray, p: QualityParams) -> tuple[float, str | None]:
    """(quality 0..1, причина отказа или None). face — объект InsightFace (det_score, bbox, kps)."""
    x1, y1, x2, y2 = [float(v) for v in face.bbox]
    side = min(x2 - x1, y2 - y1)
    if float(face.det_score) < p.min_det_score:
        return 0.0, "low_det_score"
    if side < p.min_side_px:
        return 0.0, "too_small"
    h, w = image.shape[:2]
    roi = image[max(0, int(y1)):min(h, int(y2)), max(0, int(x1)):min(w, int(x2))]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if roi.ndim == 3 and roi.size else roi
    s_det = float(face.det_score)
    s_size = min(side / p.good_side_px, 1.0)
    s_sharp = min(sharpness(gray) / p.good_sharpness, 1.0)
    s_front = frontality(face.kps)
    q = p.w_det * s_det + p.w_size * s_size + p.w_sharp * s_sharp + p.w_frontal * s_front
    return round(float(q), 3), None
