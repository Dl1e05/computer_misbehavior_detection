"""Огонь: событие, только если пламя видно на нескольких прогонах подряд в одном месте.

- Модель огня гоняется не каждый кадр, а run_hz раз в секунду (решает конвейер);
  правило получает ctx.fire только на кадрах, где модель запускалась (иначе None).
- Временной фильтр (ТЗ): пламя найдено в >= min_hits из последних window_runs прогонов
  примерно в одном месте (старт: 4 из 5).
- Одно событие на очаг: повтор — только если огня не было clear_s секунд.
- Опция flicker (ТЗ 6.3, блики): настоящее пламя «дрожит» — меняются размер рамки и
  яркость; статичная лампа — нет. Включать по результатам тестов.
- participants пустой, вместо них bbox очага (контракт шага 8).
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime

import cv2
import numpy as np

from vision.rules.base import Detection, Rule, RuleContext, register


@dataclass
class _Run:
    ts: datetime
    boxes: list[tuple[float, float, float, float, float]]   # x1, y1, x2, y2 (доли 0–1), conf
    brightness: list[float]                                  # средняя яркость в рамке, 0–1


def _iou(a, b) -> float:
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def same_place(a, b, iou_min: float = 0.1, center_k: float = 0.5) -> bool:
    """Рамки примерно в одном месте: пересекаются или центры близко (доля размера рамки)."""
    if _iou(a, b) >= iou_min:
        return True
    ca = ((a[0] + a[2]) / 2, (a[1] + a[3]) / 2)
    cb = ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
    size = max(a[2] - a[0], a[3] - a[1], b[2] - b[0], b[3] - b[1])
    return abs(ca[0] - cb[0]) <= center_k * size and abs(ca[1] - cb[1]) <= center_k * size


@register("fire")
class FireRule(Rule):
    incident_type = "fire"
    requires = {"fire"}

    def __init__(self, camera_id: str, params: dict):
        super().__init__(camera_id, params)
        p = params
        self.window_runs = int(p.get("window_runs", 5))
        self.min_hits = int(p.get("min_hits", 4))
        self.clear_s = float(p.get("clear_s", 10.0))
        self.classes = set(p.get("classes", ["fire"]))        # можно ["fire", "smoke"]
        self.min_conf = float(p.get("min_conf", 0.25))
        self.flicker = bool(p.get("flicker", False))
        self.flicker_min = float(p.get("flicker_min", 0.05))  # мин. разброс размера/яркости
        self._runs: deque[_Run] = deque(maxlen=self.window_runs)
        self._active = False
        self._last_fire: datetime | None = None

    def reset(self) -> None:
        self._runs.clear()
        self._active = False
        self._last_fire = None

    def update(self, ctx: RuleContext) -> list[Detection]:
        if ctx.fire is None:                       # на этом кадре модель огня не запускалась
            return []
        names = getattr(ctx, "fire_names", None) or {0: "smoke", 1: "fire"}
        boxes, bright = [], []
        gray = cv2.cvtColor(ctx.image, cv2.COLOR_BGR2GRAY) if self.flicker and ctx.image is not None else None
        for x1, y1, x2, y2, conf, cls in np.asarray(ctx.fire).reshape(-1, 6):
            if conf < self.min_conf or names.get(int(cls), str(int(cls))) not in self.classes:
                continue
            nb = ctx.norm_bbox((x1, y1, x2, y2))
            boxes.append((*nb, float(conf)))
            if gray is not None:
                crop = gray[int(y1):int(y2) + 1, int(x1):int(x2) + 1]
                bright.append(float(crop.mean()) / 255 if crop.size else 0.0)
        self._runs.append(_Run(ctx.ts, boxes, bright))
        if boxes:
            self._last_fire = ctx.ts

        if self._active:                           # очаг уже сообщён — ждём, пока погаснет
            if self._last_fire and (ctx.ts - self._last_fire).total_seconds() >= self.clear_s:
                self._active = False
            return []

        if not boxes:
            return []
        ref = max(boxes, key=lambda b: b[4])       # самая уверенная рамка — кандидат в очаг
        matched = []                               # (run_idx, box, brightness)
        for run in self._runs:
            for i, b in enumerate(run.boxes):
                if same_place(ref[:4], b[:4]):
                    matched.append((b, run.brightness[i] if i < len(run.brightness) else None))
                    break
        if len(matched) < self.min_hits:
            return []
        if self.flicker and not self._flickers(matched):
            return []

        self._active = True
        conf = float(np.mean([b[4] for b, _ in matched]))
        return [Detection(
            incident_type=self.incident_type, event_type="detection", camera_id=self.camera_id,
            occurred_at=ctx.ts, confidence=round(conf, 3), participants=[],
            bbox=tuple(round(v, 4) for v in ref[:4]),
        )]

    def _flickers(self, matched) -> bool:
        """Пламя меняет размер и/или яркость между прогонами; лампа/экран — почти нет."""
        areas = np.array([(b[2] - b[0]) * (b[3] - b[1]) for b, _ in matched])
        area_var = float(areas.std() / areas.mean()) if areas.mean() > 0 else 0.0
        br = np.array([v for _, v in matched if v is not None])
        bright_var = float(br.std()) if br.size >= 2 else 0.0
        return max(area_var, bright_var) >= self.flicker_min
