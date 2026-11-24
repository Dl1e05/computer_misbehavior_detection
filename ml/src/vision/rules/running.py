"""Бег (ТЗ 18): скорость человека в «ростах в секунду».

Скорость = смещение ног за window_s / время / медианная высота рамки за это окно.
Рост как масштаб убирает перспективу: дальний человек мелкий и «медленный» в пикселях,
но в ростах — такой же. Ориентиры: шаг ~0.7–1.0 роста/с, бег ~1.5–3.
- бег засчитывается, если скорость >= speed этой длительности min_s подряд
  (один рывок рамки при перекрытии — не бег);
- одно событие на трек; снова — после rearm_s спокойного движения.
Событие: incident running, участник — трек, confidence — уверенность трека.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

from vision.rules.base import Detection, Participant, Rule, RuleContext, register


@dataclass
class _State:
    hist: deque = field(default_factory=deque)        # (ts, x, y, h) в пикселях
    fast_since: datetime | None = None
    calm_since: datetime | None = None
    alerted: bool = False
    last_seen: datetime | None = None
    speed: float = 0.0


@register("running")
class RunningRule(Rule):
    incident_type = "running"
    requires = {"persons"}

    def __init__(self, camera_id: str, params: dict):
        super().__init__(camera_id, params)
        self.speed = float(params.get("speed", 1.6))
        self.window_s = float(params.get("window_s", 1.0))
        self.min_s = float(params.get("min_s", 1.0))
        self.rearm_s = float(params.get("rearm_s", 5.0))
        self.forget_s = float(params.get("forget_s", 5.0))
        self._state: dict[int, _State] = {}

    def reset(self) -> None:
        self._state.clear()

    def speed_of(self, track_id: int) -> float | None:
        st = self._state.get(track_id)
        return st.speed if st else None

    def update(self, ctx: RuleContext) -> list[Detection]:
        now = ctx.ts
        out = []
        for t in ctx.tracks:
            st = self._state.setdefault(t.track_id, _State())
            st.last_seen = now
            x1, y1, x2, y2 = t.bbox
            st.hist.append((now, (x1 + x2) / 2, y2, y2 - y1))
            while st.hist and (now - st.hist[0][0]).total_seconds() > self.window_s:
                st.hist.popleft()
            dt = (now - st.hist[0][0]).total_seconds()
            if dt < 0.6 * self.window_s:                      # окно ещё не набралось
                continue
            _, xa, ya, _ = st.hist[0]
            h = float(np.median([s[3] for s in st.hist]))
            st.speed = float(np.hypot(st.hist[-1][1] - xa, st.hist[-1][2] - ya) / dt / max(h, 1.0))
            if st.speed >= self.speed:
                st.calm_since = None
                st.fast_since = st.fast_since or now
                if not st.alerted and (now - st.fast_since).total_seconds() >= self.min_s:
                    st.alerted = True
                    out.append(Detection(
                        incident_type=self.incident_type, event_type="detection", camera_id=self.camera_id,
                        occurred_at=now, confidence=round(t.conf, 3),
                        participants=[Participant(t.track_id, ctx.norm_bbox(t.bbox), round(t.conf, 3))],
                        extra={"speed": round(st.speed, 2)}))
            else:
                st.fast_since = None
                if st.alerted:
                    st.calm_since = st.calm_since or now
                    if (now - st.calm_since).total_seconds() >= self.rearm_s:
                        st.alerted, st.calm_since = False, None
        for tid, st in list(self._state.items()):
            if st.last_seen and (now - st.last_seen).total_seconds() > self.forget_s:
                del self._state[tid]
        return out
