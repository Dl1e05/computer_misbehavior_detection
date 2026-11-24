"""Скопление людей (ТЗ 18): в зоне не меньше min_count человек дольше min_s.

- считаем треки, чьи ноги внутри зоны; нужна вся площадь кадра — нарисуй зону на весь кадр;
- одно событие на скопление; снова — когда людей стало меньше min_count на clear_s;
- расписание зоны учитывается.
Порог min_count — свой для каждой камеры (холл и узкий коридор — разные).
Событие: incident crowding, участники — до max_participants треков в зоне, zone_id, extra: count.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from vision.analytics.zones import Zone, foot_point
from vision.rules.base import Detection, Participant, Rule, RuleContext, register


@dataclass
class _State:
    above_since: datetime | None = None
    below_since: datetime | None = None
    alerted: bool = False


@register("crowding")
class CrowdingRule(Rule):
    incident_type = "crowding"
    requires = {"persons"}
    uses_zones = True

    def __init__(self, camera_id: str, params: dict):
        super().__init__(camera_id, params)
        self.min_count = int(params.get("min_count", 8))
        self.min_s = float(params.get("min_s", 5.0))
        self.clear_s = float(params.get("clear_s", 10.0))
        self.max_participants = int(params.get("max_participants", 20))
        self.zones: dict[str, Zone] = {z.zone_id: z for z in params.get("zones", [])
                                       if isinstance(z, Zone) and z.active and z.camera_id == camera_id}
        self._state: dict[str, _State] = {}

    def reset(self) -> None:
        self._state.clear()

    def update(self, ctx: RuleContext) -> list[Detection]:
        now = ctx.ts
        out = []
        for zone in self.zones.values():
            st = self._state.setdefault(zone.zone_id, _State())
            if not zone.armed(now):
                self._state[zone.zone_id] = _State()
                continue
            inside = [t for t in ctx.tracks if zone.contains(*foot_point(t.bbox, ctx.frame_w, ctx.frame_h))]
            if len(inside) >= self.min_count:
                st.below_since = None
                st.above_since = st.above_since or now
                if not st.alerted and (now - st.above_since).total_seconds() >= self.min_s:
                    st.alerted = True
                    top = sorted(inside, key=lambda t: -t.conf)[:self.max_participants]
                    out.append(Detection(
                        incident_type=self.incident_type, event_type="detection", camera_id=self.camera_id,
                        occurred_at=now, confidence=round(min(1.0, len(inside) / (self.min_count * 1.5)), 3),
                        participants=[Participant(t.track_id, ctx.norm_bbox(t.bbox), round(t.conf, 3)) for t in top],
                        zone_id=zone.zone_id, zone_version=zone.version, extra={"count": len(inside)}))
            else:
                st.above_since = None
                if st.alerted:
                    st.below_since = st.below_since or now
                    if (now - st.below_since).total_seconds() >= self.clear_s:
                        st.alerted, st.below_since = False, None
        return out
