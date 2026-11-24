"""Долгое нахождение в зоне (ТЗ 18): человек стоит/ходит в зоне дольше dwell_s.

- точка человека — ноги (середина нижней границы рамки);
- короткие выходы за границу (до gap_s) не обнуляют время — человек топчется у края;
- одно событие на трек и зону; снова — только после выхода дольше gap_s;
- расписание зоны учитывается (вне окна зона не охраняется).
Ограничение: время считается по track_id — если трекер сменил ID, отсчёт начнётся заново.
Событие: incident loitering, участник — трек, zone_id/zone_version, extra: dwell_s.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from vision.analytics.zones import Zone, foot_point
from vision.rules.base import Detection, Participant, Rule, RuleContext, register


@dataclass
class _State:
    since: datetime | None = None        # с какого момента в зоне (с учётом коротких выходов)
    last_inside: datetime | None = None
    alerted: bool = False


@register("loitering")
class LoiteringRule(Rule):
    incident_type = "loitering"
    requires = {"persons"}
    uses_zones = True

    def __init__(self, camera_id: str, params: dict):
        super().__init__(camera_id, params)
        self.dwell_s = float(params.get("dwell_s", 60.0))
        self.gap_s = float(params.get("gap_s", 3.0))
        self.zones: dict[str, Zone] = {z.zone_id: z for z in params.get("zones", [])
                                       if isinstance(z, Zone) and z.active and z.camera_id == camera_id}
        self._state: dict[tuple[int, str], _State] = {}

    def reset(self) -> None:
        self._state.clear()

    def update(self, ctx: RuleContext) -> list[Detection]:
        now = ctx.ts
        out = []
        for zone in self.zones.values():
            if not zone.armed(now):
                for key in [k for k in self._state if k[1] == zone.zone_id]:
                    del self._state[key]
                continue
            for t in ctx.tracks:
                if not zone.contains(*foot_point(t.bbox, ctx.frame_w, ctx.frame_h)):
                    continue
                st = self._state.setdefault((t.track_id, zone.zone_id), _State())
                if st.since is None or (now - st.last_inside).total_seconds() > self.gap_s:
                    st.since, st.alerted = now, False           # вошёл (или вернулся после долгого выхода)
                st.last_inside = now
                dwell = (now - st.since).total_seconds()
                if dwell >= self.dwell_s and not st.alerted:
                    st.alerted = True
                    out.append(Detection(
                        incident_type=self.incident_type, event_type="detection", camera_id=self.camera_id,
                        occurred_at=now, confidence=round(t.conf, 3),
                        participants=[Participant(t.track_id, ctx.norm_bbox(t.bbox), round(t.conf, 3))],
                        zone_id=zone.zone_id, zone_version=zone.version, extra={"dwell_s": round(dwell, 1)}))
        for key, st in list(self._state.items()):                 # давно не в зоне — забыть
            if st.last_inside and (now - st.last_inside).total_seconds() > max(self.gap_s, 10.0):
                del self._state[key]
        return out
