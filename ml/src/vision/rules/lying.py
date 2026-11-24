"""Человек лежит без движения (ТЗ 18) — правило падения без фазы «упал».

Ловит то, что fall пропускает: человек медленно сел и лёг (обморок у стены), упал вне кадра,
или падение не распозналось. Поэтому порог времени больше, чем у fall (по умолчанию 30 с).
- поза горизонтальная (торс от вертикали > torso_angle_deg или рамка шире высоты) — те же
  признаки, что у fall;
- неподвижен: точки скелета (или, если скелета нет, центр рамки) смещаются меньше still_motion
  размера человека за кадр;
- лежит и неподвижен lie_s подряд -> одно событие; встал (не горизонтален get_up_s) — сброс.
Без скелета решение по рамке, уверенность события ниже (x0.7).
Лежать на лавке тоже «лежит без движения» — включать на камерах, где так быть не должно
(коридор, лестница), а не в зоне отдыха.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np

from vision.rules.base import Detection, Participant, RuleContext, register
from vision.rules.fall import FallRule


@dataclass
class _State:
    lying_since: datetime | None = None
    still_since: datetime | None = None
    upright_since: datetime | None = None
    prev_kpts: np.ndarray | None = None
    prev_center: tuple[float, float] | None = None
    alerted: bool = False
    last_seen: datetime | None = None
    had_pose: bool = False


@register("lying_motionless")
class LyingMotionlessRule(FallRule):
    incident_type = "lying_motionless"
    requires = {"persons", "poses"}

    def __init__(self, camera_id: str, params: dict):
        super().__init__(camera_id, params)
        self.lie_s = float(params.get("lie_s", 30.0))
        self._lie: dict[int, _State] = {}

    def reset(self) -> None:
        self._lie.clear()

    def phase_of(self, track_id: int) -> str | None:
        st = self._lie.get(track_id)
        return "lying" if st and st.lying_since else None

    def update(self, ctx: RuleContext) -> list[Detection]:
        now = ctx.ts
        poses = ctx.poses or {}
        out = []
        for t in ctx.tracks:
            st = self._lie.setdefault(t.track_id, _State())
            st.last_seen = now
            kpts = poses.get(t.track_id)
            _, horizontal, _ = self._features(t.bbox, kpts)
            x1, y1, x2, y2 = t.bbox
            center = ((x1 + x2) / 2, (y1 + y2) / 2)
            size = max(x2 - x1, y2 - y1, 1e-6)
            motion = self._motion(st.prev_kpts, kpts, t.bbox)
            if motion is None and st.prev_center is not None:          # скелета нет — по центру рамки
                motion = float(np.hypot(center[0] - st.prev_center[0], center[1] - st.prev_center[1]) / size)
            st.prev_kpts, st.prev_center = kpts, center
            if horizontal:
                st.upright_since = None
                st.lying_since = st.lying_since or now
                st.had_pose = st.had_pose or kpts is not None
                if st.still_since is None or (motion is not None and motion > self.still_motion):
                    st.still_since = now
                still = (now - st.still_since).total_seconds()
                if not st.alerted and still >= self.lie_s:
                    st.alerted = True
                    conf = t.conf if st.had_pose else t.conf * 0.7
                    out.append(Detection(
                        incident_type=self.incident_type, event_type="detection", camera_id=self.camera_id,
                        occurred_at=now, confidence=round(float(conf), 3),
                        participants=[Participant(t.track_id, ctx.norm_bbox(t.bbox), round(t.conf, 3))]))
            else:
                st.upright_since = st.upright_since or now
                if (now - st.upright_since).total_seconds() >= self.get_up_s:
                    self._lie[t.track_id] = _State(last_seen=now)          # встал — всё заново
        for tid, st in list(self._lie.items()):
            if st.last_seen and (now - st.last_seen).total_seconds() > self.forget_s:
                del self._lie[tid]
        return out
