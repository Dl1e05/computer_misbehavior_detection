"""Оценка «насколько это похоже на драку» для группы людей за окно ~2 с.

Интерфейс FightScorer один, реализации меняются строкой конфига (scorer: ...):
- heuristic   — сейчас: близость + резкие движения рук по скелетам;
- skeleton    — потом (Фаза 3.2, ступень 1): GRU/1D-свёртка по скелетам группы;
- video_model — потом (ступень 2): r2plus1d_18 по кропу вокруг группы.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Protocol

import numpy as np

from vision.inference.pose import L_ELBOW, L_HIP, L_SHOULDER, L_WRIST, R_ELBOW, R_HIP, R_SHOULDER, R_WRIST

ARM_POINTS = (L_ELBOW, R_ELBOW, L_WRIST, R_WRIST)
TORSO_POINTS = (L_SHOULDER, R_SHOULDER, L_HIP, R_HIP)


@dataclass
class FrameObs:
    """Что видно на одном кадре камеры: рамки треков (пиксели) и их скелеты."""
    ts: datetime
    boxes: dict[int, tuple[float, float, float, float]]
    poses: dict[int, np.ndarray] = field(default_factory=dict)   # (17, 3)
    image: Any = None                                             # только если scorer.needs_images


class FightScorer(Protocol):
    needs_images: bool

    def score(self, window: list[FrameObs], group: list[int], window_s: float | None = None) -> float:
        """0..1 — уверенность, что группа дерётся в этом окне (window_s — полная длина окна)."""
        ...


def close(a, b, k: float = 1.0) -> bool:
    """Рамки пересекаются или центры ближе k * роста (бо́льшей из двух рамок)."""
    if a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]:
        return True
    ca = ((a[0] + a[2]) / 2, (a[1] + a[3]) / 2)
    cb = ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
    h = max(a[3] - a[1], b[3] - b[1])
    return math.dist(ca, cb) < k * h


def find_groups(boxes: dict[int, tuple], k: float = 1.0) -> list[list[int]]:
    """Группы из 2+ человек, связанных близостью (связные компоненты)."""
    ids = list(boxes)
    parent = {i: i for i in ids}

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            if close(boxes[a], boxes[b], k):
                parent[root(a)] = root(b)
    groups: dict[int, list[int]] = {}
    for i in ids:
        groups.setdefault(root(i), []).append(i)
    return [sorted(g) for g in groups.values() if len(g) >= 2]


class HeuristicFightScorer:
    """score = доля кадров окна, где группа рядом И у кого-то резко движутся руки.

    Скорость точек рук считается ОТНОСИТЕЛЬНО КОРПУСА (центр плеч и бёдер) в долях
    роста, в «ростах в секунду»: ходьба и приближение к камере вычитаются, остаются
    только движения рук. Не зависит от разрешения и FPS. «Резко» = быстро, или средне быстро
    со сменой направления (замах -> удар). Просто стоять рядом — не драка (ТЗ 6.3).
    """
    needs_images = False

    def __init__(self, close_k: float = 1.0, arm_speed: float = 1.5,
                 turn_speed: float = 0.8, turn_deg: float = 90.0, kpt_conf: float = 0.5):
        self.close_k = close_k
        self.arm_speed = arm_speed        # ростов/с — «быстро»
        self.turn_speed = turn_speed      # ростов/с — «средне», если со сменой направления
        self.turn_deg = turn_deg
        self.kpt_conf = kpt_conf

    def _body_frame(self, k: np.ndarray, box) -> tuple[np.ndarray, float] | None:
        """Центр корпуса и рост — система координат, в которой меряем руки."""
        vis = [i for i in TORSO_POINTS if k[i, 2] > self.kpt_conf]
        if len(vis) < 2:
            return None
        return k[vis, :2].mean(axis=0), max(box[3] - box[1], 1e-6)

    def _arm_velocities(self, a: FrameObs, b: FrameObs, tid: int) -> list[np.ndarray]:
        """Скорости видимых точек рук относительно корпуса между кадрами a и b, в ростах/с."""
        ka, kb = a.poses.get(tid), b.poses.get(tid)
        box_a, box_b = a.boxes.get(tid), b.boxes.get(tid)
        dt = (b.ts - a.ts).total_seconds()
        if ka is None or kb is None or box_a is None or box_b is None or dt <= 0:
            return []
        fa, fb = self._body_frame(ka, box_a), self._body_frame(kb, box_b)
        if fa is None or fb is None:
            return []
        out = []
        for i in ARM_POINTS:
            if ka[i, 2] > self.kpt_conf and kb[i, 2] > self.kpt_conf:
                ra = (ka[i, :2] - fa[0]) / fa[1]          # рука в координатах корпуса, доли роста
                rb = (kb[i, :2] - fb[0]) / fb[1]
                out.append((rb - ra) / dt)
            else:
                out.append(None)
        return out

    def _sharp(self, v_prev: list, v_cur: list) -> bool:
        for p, c in zip(v_prev or [None] * 4, v_cur):
            if c is None:
                continue
            speed = float(np.linalg.norm(c))
            if speed >= self.arm_speed:
                return True
            if p is not None and speed >= self.turn_speed and np.linalg.norm(p) >= self.turn_speed:
                cos = float(np.dot(p, c) / (np.linalg.norm(p) * speed))
                if math.degrees(math.acos(max(-1.0, min(1.0, cos)))) >= self.turn_deg:
                    return True
        return False

    def score(self, window: list[FrameObs], group: list[int], window_s: float | None = None) -> float:
        """Доля «драчных» шагов от ВСЕГО окна: кадры, где группы ещё не было, — не драка.
        window_s — полная длина окна: пока окно не заполнилось, score не завышается."""
        if len(window) < 3:
            return 0.0
        hits = 0
        prev_v: dict[int, list] = {}
        for a, b in zip(window, window[1:]):
            present = [t for t in group if t in b.boxes]
            if len(present) < 2:
                prev_v = {}
                continue
            near = any(close(b.boxes[x], b.boxes[y], self.close_k)
                       for i, x in enumerate(present) for y in present[i + 1:])
            sharp = False
            for t in present:
                v = self._arm_velocities(a, b, t)
                if v and self._sharp(prev_v.get(t), v):
                    sharp = True
                prev_v[t] = v
            if near and sharp:
                hits += 1
        steps = len(window) - 1
        if window_s:
            dt = (window[-1].ts - window[0].ts).total_seconds() / steps
            steps = max(steps, round(window_s / dt) if dt > 0 else steps)
        return min(1.0, hits / steps)


def make_scorer(name: str, params: dict | None = None) -> FightScorer:
    params = params or {}
    if name == "heuristic":
        return HeuristicFightScorer(**params)
    raise ValueError(f"FightScorer '{name}' пока не реализован (есть: heuristic; "
                     f"skeleton и video_model — после обучения, Фаза 3.2)")
