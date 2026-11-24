"""Вход в запретную зону — чистая геометрия, без обучения.

Логика для каждой пары (трек, зона):
- точка человека = середина нижней границы рамки (где ноги);
- вход засчитывается, когда трек пробыл внутри min_inside_s подряд
  (антидребезг: шаг на границу и обратно — не вход);
- появился сразу внутри — тоже вход, если spawn_inside_counts=true;
- одно событие на трек и зону, пока он не выйдет: выход засчитывается
  только после min_outside_s снаружи (иначе дрожание рамки на границе
  дало бы серию событий). Объединение повторов за 60 с делает BE.

Расписание зоны (zone.schedule, например «после 22:00»): вне окна зона не охраняется и
состояние сбрасывается; когда окно началось, кто уже внутри — считается «появившимся внутри».

Пороги в секундах, а не в кадрах — смена process_fps ничего не ломает.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from vision.analytics.zones import Zone, foot_point
from vision.rules.base import Detection, Participant, Rule, RuleContext, register


@dataclass
class _State:
    seen_outside: bool = False             # трек был замечен вне зоны
    inside_since: datetime | None = None   # с какого момента непрерывно внутри
    outside_since: datetime | None = None  # с какого момента снаружи (после события)
    alerted: bool = False                  # событие по этому входу уже было
    last_seen: datetime | None = None


@register("zone_intrusion")
class ZoneIntrusionRule(Rule):
    incident_type = "zone_intrusion"
    requires = {"persons"}
    uses_zones = True

    def __init__(self, camera_id: str, params: dict):
        super().__init__(camera_id, params)
        self.min_inside_s = float(params.get("min_inside_s", 0.5))
        self.min_outside_s = float(params.get("min_outside_s", 1.0))
        self.spawn_inside_counts = bool(params.get("spawn_inside_counts", True))
        self.forget_s = float(params.get("forget_s", 10.0))   # трек пропал — забыть состояние
        self.zones: dict[str, Zone] = {}
        self._state: dict[tuple[int, str], _State] = {}
        self.set_zones(params.get("zones", []))

    def set_zones(self, zones: list[Zone]) -> None:
        """Обновление зон (из API BE раз в 30 с). Изменилась версия — состояние зоны заново."""
        new = {z.zone_id: z for z in zones
               if isinstance(z, Zone) and z.active and z.camera_id == self.camera_id}
        for key in list(self._state):
            old, cur = self.zones.get(key[1]), new.get(key[1])
            if cur is None or old is None or old.version != cur.version:
                del self._state[key]
        self.zones = new

    def reset(self) -> None:
        self._state.clear()

    def update(self, ctx: RuleContext) -> list[Detection]:
        now = ctx.ts
        out: list[Detection] = []
        for zone in self.zones.values():
            if not zone.armed(now):                     # по расписанию зона сейчас не охраняется
                for key in [k for k in self._state if k[1] == zone.zone_id]:
                    del self._state[key]                # включится — кто внутри, тот «появился внутри»
                continue
            entered, inside = [], []
            for t in ctx.tracks:
                st = self._state.setdefault((t.track_id, zone.zone_id), _State())
                st.last_seen = now
                if zone.contains(*foot_point(t.bbox, ctx.frame_w, ctx.frame_h)):
                    st.outside_since = None
                    if st.inside_since is None:
                        st.inside_since = now
                    inside.append(t)
                    long_enough = (now - st.inside_since).total_seconds() >= self.min_inside_s
                    allowed = st.seen_outside or self.spawn_inside_counts
                    if long_enough and allowed and not st.alerted:
                        st.alerted = True
                        entered.append(t)
                else:
                    st.inside_since = None
                    st.seen_outside = True
                    if st.alerted:
                        if st.outside_since is None:
                            st.outside_since = now
                        if (now - st.outside_since).total_seconds() >= self.min_outside_s:
                            st.alerted = False          # вышел по-настоящему — следующий вход снова событие
                            st.outside_since = None
            if entered:
                ids = {t.track_id for t in entered}
                others = [t for t in inside if t.track_id not in ids]
                out.append(Detection(
                    incident_type=self.incident_type,
                    event_type="detection",
                    camera_id=self.camera_id,
                    occurred_at=now,
                    confidence=round(max(t.conf for t in entered), 3),
                    participants=[Participant(t.track_id, ctx.norm_bbox(t.bbox), round(t.conf, 3))
                                  for t in entered + others],
                    zone_id=zone.zone_id,
                    zone_version=zone.version,
                ))
        self._forget(now)
        return out

    def _forget(self, now: datetime) -> None:
        for key, st in list(self._state.items()):
            if st.last_seen and (now - st.last_seen).total_seconds() > self.forget_s:
                del self._state[key]
