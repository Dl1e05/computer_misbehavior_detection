"""Падение: быстрый переход в горизонтальную позу И неподвижность N секунд.

Одна горизонтальная поза — не падение (ТЗ 6.3): человек может лежать на лавке.
Машина состояний на каждый track_id:

  UPRIGHT  -> FALLING   за <= fall_window_s центр бёдер опустился > drop_ratio высоты
                        рамки (высота — до падения) И поза горизонтальная
  FALLING  -> ON_GROUND поза остаётся горизонтальной (не поднялся за fall_window_s)
  ON_GROUND             неподвижен still_s подряд -> событие fall (одно на падение)
  ON_GROUND -> UPRIGHT  встал (не горизонтален get_up_s) — события нет / можно снова

Горизонтальная поза: торс (середина плеч -> середина бёдер) отклонён от вертикали
> torso_angle_deg, либо ширина рамки / высота > aspect_ratio.
Учитываются только точки с conf > kpt_conf. Не видно плеч или бёдер — переход в
FALLING не засчитывается (по ТЗ 5.2 лучше пропустить, чем выдумать падение).
Но в кадрах, где скелет пропал (смаз в момент падения — частый случай), высоту
бёдер для ИСТОРИИ оцениваем по центру рамки — иначе «до падения» не с чем сравнить.

Все числа — стартовые для настройки (в cameras.yaml), не доказанные значения.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

from vision.inference.pose import L_HIP, L_SHOULDER, R_HIP, R_SHOULDER
from vision.rules.base import Detection, Participant, Rule, RuleContext, register

UPRIGHT, FALLING, ON_GROUND = "upright", "falling", "on_ground"


@dataclass
class _Sample:
    ts: datetime
    hip_y: float | None
    height: float
    horizontal: bool


@dataclass
class _State:
    phase: str = UPRIGHT
    history: deque = field(default_factory=deque)     # _Sample за последние fall_window_s
    phase_since: datetime | None = None
    still_since: datetime | None = None
    upright_since: datetime | None = None             # для get_up_s в ON_GROUND
    prev_kpts: np.ndarray | None = None
    alerted: bool = False
    torso_conf: float = 0.0
    last_seen: datetime | None = None


def _mid(kpts: np.ndarray, a: int, b: int, thr: float) -> tuple[float, float, float] | None:
    """Середина двух точек (или одна видимая). (x, y, средняя conf) либо None."""
    pts = [kpts[i] for i in (a, b) if kpts[i, 2] > thr]
    if not pts:
        return None
    return (float(np.mean([p[0] for p in pts])), float(np.mean([p[1] for p in pts])),
            float(np.mean([p[2] for p in pts])))


@register("fall")
class FallRule(Rule):
    incident_type = "fall"
    requires = {"persons", "poses"}

    def __init__(self, camera_id: str, params: dict):
        super().__init__(camera_id, params)
        p = params
        self.fall_window_s = float(p.get("fall_window_s", 1.0))
        self.drop_ratio = float(p.get("drop_ratio", 0.3))
        self.torso_angle_deg = float(p.get("torso_angle_deg", 60.0))
        self.aspect_ratio = float(p.get("aspect_ratio", 1.0))
        self.still_s = float(p.get("still_s", 3.0))
        self.still_motion = float(p.get("still_motion", 0.05))   # доля размера человека за кадр
        self.get_up_s = float(p.get("get_up_s", 0.5))
        self.kpt_conf = float(p.get("kpt_conf", 0.5))
        self.forget_s = float(p.get("forget_s", 10.0))
        self._state: dict[int, _State] = {}

    def reset(self) -> None:
        self._state.clear()

    # ---------- признаки позы ----------

    def _features(self, bbox, kpts: np.ndarray | None):
        """-> (hip_y | None, горизонтален?, уверенность по торсу)."""
        x1, y1, x2, y2 = bbox
        w, h = max(x2 - x1, 1e-6), max(y2 - y1, 1e-6)
        wide = w / h > self.aspect_ratio
        if kpts is None:
            return None, wide, 0.0
        sh = _mid(kpts, L_SHOULDER, R_SHOULDER, self.kpt_conf)
        hp = _mid(kpts, L_HIP, R_HIP, self.kpt_conf)
        if hp is None:
            return None, wide, 0.0
        if sh is None:
            return hp[1], wide, hp[2] / 2
        dx, dy = hp[0] - sh[0], hp[1] - sh[1]
        angle = math.degrees(math.atan2(abs(dx), abs(dy)))        # 0 — стоит, 90 — лежит
        return hp[1], wide or angle > self.torso_angle_deg, (sh[2] + hp[2]) / 2

    def _motion(self, prev: np.ndarray | None, cur: np.ndarray | None, bbox) -> float | None:
        """Среднее смещение видимых точек между кадрами, в долях размера человека."""
        if prev is None or cur is None:
            return None
        vis = (prev[:, 2] > self.kpt_conf) & (cur[:, 2] > self.kpt_conf)
        if vis.sum() < 3:
            return None
        size = max(bbox[2] - bbox[0], bbox[3] - bbox[1], 1e-6)   # лежит — значит ширина
        return float(np.linalg.norm(cur[vis, :2] - prev[vis, :2], axis=1).mean() / size)

    # ---------- основной цикл ----------

    def update(self, ctx: RuleContext) -> list[Detection]:
        now = ctx.ts
        poses = ctx.poses or {}
        out = []
        for t in ctx.tracks:
            st = self._state.setdefault(t.track_id, _State(phase_since=now))
            st.last_seen = now
            kpts = poses.get(t.track_id)
            hip_y, horizontal, torso_conf = self._features(t.bbox, kpts)
            height = t.bbox[3] - t.bbox[1]

            hist_hip = hip_y if hip_y is not None else (t.bbox[1] + t.bbox[3]) / 2   # скелета нет — центр рамки
            st.history.append(_Sample(now, hist_hip, height, horizontal))
            while st.history and (now - st.history[0].ts).total_seconds() > self.fall_window_s:
                st.history.popleft()

            if st.phase == UPRIGHT:
                if horizontal and hip_y is not None and self._dropped(st, hip_y):
                    self._go(st, FALLING, now)
                    st.torso_conf = torso_conf
            elif st.phase == FALLING:
                if horizontal:
                    self._go(st, ON_GROUND, now)
                    st.still_since = now
                elif (now - st.phase_since).total_seconds() > self.fall_window_s:
                    self._go(st, UPRIGHT, now)
            elif st.phase == ON_GROUND:
                if not horizontal:
                    st.upright_since = st.upright_since or now
                    if (now - st.upright_since).total_seconds() >= self.get_up_s:
                        self._go(st, UPRIGHT, now)                 # встал — событие не нужно
                        st.alerted = False
                        st.history.clear()
                else:
                    st.upright_since = None
                    m = self._motion(st.prev_kpts, kpts, t.bbox)
                    if m is not None and m > self.still_motion:
                        st.still_since = now                       # шевелится — ждём заново
                    if (not st.alerted and st.still_since is not None
                            and (now - st.still_since).total_seconds() >= self.still_s):
                        st.alerted = True
                        conf = round(float(min(t.conf, st.torso_conf or t.conf)), 3)
                        out.append(Detection(
                            incident_type=self.incident_type, event_type="detection",
                            camera_id=self.camera_id, occurred_at=now, confidence=conf,
                            participants=[Participant(t.track_id, ctx.norm_bbox(t.bbox), round(t.conf, 3))],
                        ))
            st.prev_kpts = kpts
        self._forget(now)
        return out

    def _dropped(self, st: _State, hip_y: float) -> bool:
        """Бёдра опустились больше чем на drop_ratio высоты (высота — когда ещё стоял)."""
        for s in st.history:
            if s.hip_y is not None and not s.horizontal and s.height > 0:
                if hip_y - s.hip_y > self.drop_ratio * s.height:
                    return True
        return False

    @staticmethod
    def _go(st: _State, phase: str, now: datetime) -> None:
        st.phase, st.phase_since = phase, now
        st.upright_since = None

    def phase_of(self, track_id: int) -> str | None:
        """Для отрисовки/отладки."""
        st = self._state.get(track_id)
        return st.phase if st else None

    def _forget(self, now: datetime) -> None:
        for tid, st in list(self._state.items()):
            if st.last_seen and (now - st.last_seen).total_seconds() > self.forget_s:
                del self._state[tid]
