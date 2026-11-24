"""Пересечение линии с направлением (ТЗ 18) — основа для подсчёта проходов и «прохода вдвоём».

Точка человека — середина нижней границы рамки (ноги). Для каждой пары (трек, линия):
- полоса band по обе стороны линии (доля кадра) — «ничья земля»: пока ноги в ней, сторона
  не меняется (нижний край рамки прыгает, когда ноги перекрыты соседями);
- «устойчивая сторона» меняется, только если точка пробыла на новой стороне (за полосой)
  min_side_s подряд — дрожание рамки у линии не даёт серии пересечений;
- смена устойчивой стороны = пересечение, если точка в этот момент в пределах отрезка
  (с запасом margin от его длины) — обход линии сбоку не считается;
- направление: in — слева направо, out — справа налево (смотрим вдоль линии от p1 к p2);
  direction: any | in | out — какие пересечения сообщать.
Событие: incident line_crossing, участник — этот трек, extra: line_id, direction.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from vision.analytics.zones import Line, foot_point
from vision.rules.base import Detection, Participant, Rule, RuleContext, register


@dataclass
class _State:
    stable: int = 0                      # -1 слева, +1 справа, 0 — ещё не знаем
    candidate: int = 0
    candidate_since: datetime | None = None
    last_seen: datetime | None = None


@register("line_crossing")
class LineCrossingRule(Rule):
    incident_type = "line_crossing"
    requires = {"persons"}
    uses_lines = True

    def __init__(self, camera_id: str, params: dict):
        super().__init__(camera_id, params)
        self.min_side_s = float(params.get("min_side_s", 0.5))
        self.band = float(params.get("band", 0.03))
        self.direction = str(params.get("direction", "any"))
        if self.direction not in ("any", "in", "out"):
            raise ValueError(f"line_crossing: direction any | in | out, а не {self.direction}")
        self.margin = float(params.get("margin", 0.1))
        self.forget_s = float(params.get("forget_s", 10.0))
        self.lines: dict[str, Line] = {ln.line_id: ln for ln in params.get("lines", [])
                                       if isinstance(ln, Line) and ln.active and ln.camera_id == camera_id}
        self._state: dict[tuple[int, str], _State] = {}

    def reset(self) -> None:
        self._state.clear()

    def update(self, ctx: RuleContext) -> list[Detection]:
        now = ctx.ts
        out = []
        for line in self.lines.values():
            for t in ctx.tracks:
                x, y = foot_point(t.bbox, ctx.frame_w, ctx.frame_h)
                st = self._state.setdefault((t.track_id, line.line_id), _State())
                st.last_seen = now
                d = line.distance(x, y)
                if abs(d) < self.band:                    # в полосе у линии — сторону не меняем
                    if st.candidate != st.stable:
                        st.candidate, st.candidate_since = st.stable, now
                    continue
                side = 1 if d > 0 else -1
                if side != st.candidate:
                    st.candidate, st.candidate_since = side, now
                if (st.candidate != st.stable and st.candidate_since is not None
                        and (now - st.candidate_since).total_seconds() >= self.min_side_s):
                    crossed = st.stable != 0 and -self.margin <= line.along(x, y) <= 1 + self.margin
                    direction = "in" if st.candidate > 0 else "out"
                    st.stable = st.candidate
                    if crossed and self.direction in ("any", direction):
                        out.append(Detection(
                            incident_type=self.incident_type, event_type="detection", camera_id=self.camera_id,
                            occurred_at=now, confidence=round(t.conf, 3),
                            participants=[Participant(t.track_id, ctx.norm_bbox(t.bbox), round(t.conf, 3))],
                            extra={"line_id": line.line_id, "direction": direction}))
        for key, st in list(self._state.items()):
            if st.last_seen and (now - st.last_seen).total_seconds() > self.forget_s:
                del self._state[key]
        return out
