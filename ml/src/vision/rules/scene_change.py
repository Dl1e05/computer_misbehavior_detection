"""Изменение охраняемого объекта — сигнал вандализма (шаг 11, пакет D; определение отправлено BE 25.11).

Охраняемый объект — зона на кадре (дверь, окно, стенд, витрина). Событие, если:
1. вид зоны заметно отличается от эталона: доля изменённых пикселей зоны >= min_ratio, и зона
   изменилась СИЛЬНЕЕ своей окрестности (кольца вокруг) на min_ratio — если сдвинулась вся сцена
   (камеру повернули/панорама, сильная тень), это не изменение объекта (см. camera_tampering);
2. так держится hold_s подряд (не мелькание) и зону в это время НЕ заслоняет человек;
3. незадолго до начала изменения (person_window_s) человек был в зоне или рядом с ней.
Без человека рядом изменение не тревога (свет, тень, камера).
Это инцидент на проверку, не доказательство порчи. Лиц нет.

Как сравниваем (дёшево, без нейросетей; check_hz раз в секунду, кадр уменьшен до width пикселей):
- эталон зоны — медиана первых learn_s секунд, когда зону не заслоняют люди;
- кадр и эталон нормируются по яркости внутри зоны (вычесть среднее, разделить на разброс), поэтому
  включили/выключили свет или облако закрыло солнце — не изменение; меняется только «рисунок»;
- изменённый пиксель: |норм. кадр − норм. эталон| > diff_k после лёгкого размытия; пиксели под
  рамками людей не считаются (ни в зоне, ни в кольце);
- пока всё спокойно и людей нет — эталон медленно подстраивается (adapt); после события новый вид
  через relearn_s становится эталоном (одно событие на одно изменение).
Событие: incident scene_change, zone_id/zone_version, участники — кто был рядом (последняя рамка),
details: change_ratio (доля изменённой площади), seconds_since_person.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import cv2
import numpy as np

from vision.analytics.zones import Zone
from vision.rules.base import Detection, Participant, Rule, RuleContext, register


@dataclass
class _ZoneState:
    mask: np.ndarray | None = None            # маска зоны на уменьшенном кадре
    ring: np.ndarray | None = None            # окрестность зоны (кольцо шириной ~ размер зоны)
    ring_ref: np.ndarray | None = None
    learn: list = field(default_factory=list)
    learn_start: datetime | None = None
    ref: np.ndarray | None = None             # нормированный эталон (float32)
    ref_raw: np.ndarray | None = None
    change_since: datetime | None = None
    alerted_at: datetime | None = None
    last_person: dict = field(default_factory=dict)   # track_id -> (когда, рамка в долях, conf)
    ratio: float = 0.0
    excess: float = 0.0                       # насколько зона изменилась сильнее окрестности


def _norm(gray: np.ndarray, mask: np.ndarray) -> np.ndarray:
    v = gray[mask]
    return (gray - float(v.mean())) / max(float(v.std()), 5.0)


@register("scene_change")
class SceneChangeRule(Rule):
    incident_type = "scene_change"
    requires = {"persons", "image"}
    uses_zones = True

    def __init__(self, camera_id: str, params: dict):
        super().__init__(camera_id, params)
        p = params
        self.check_hz = float(p.get("check_hz", 2.0))
        self.width = int(p.get("width", 320))
        self.learn_s = float(p.get("learn_s", 5.0))
        self.min_ratio = float(p.get("min_ratio", 0.25))
        self.diff_k = float(p.get("diff_k", 1.0))
        self.hold_s = float(p.get("hold_s", 5.0))
        self.person_window_s = float(p.get("person_window_s", 60.0))
        self.near_k = float(p.get("near_k", 0.5))
        self.occlude = float(p.get("occlude", 0.1))
        self.relearn_s = float(p.get("relearn_s", 30.0))
        self.adapt = float(p.get("adapt", 0.02))
        self.ring_k = float(p.get("ring_k", 1.0))       # ширина кольца — в размерах зоны
        self.zones: dict[str, Zone] = {z.zone_id: z for z in p.get("zones", [])
                                       if isinstance(z, Zone) and z.active and z.camera_id == camera_id}
        self._state: dict[str, _ZoneState] = {}
        self._last_check: datetime | None = None

    def reset(self) -> None:
        self._state.clear()
        self._last_check = None

    def ratio_of(self, zone_id: str) -> float:
        st = self._state.get(zone_id)
        return st.ratio if st else 0.0

    def excess_of(self, zone_id: str) -> float:
        st = self._state.get(zone_id)
        return st.excess if st else 0.0

    def update(self, ctx: RuleContext) -> list[Detection]:
        now = ctx.ts
        if ctx.image is None or not self.zones:
            return []
        for zone in self.zones.values():                     # кто рядом с зоной — запоминаем каждый кадр
            st = self._state.setdefault(zone.zone_id, _ZoneState())
            for t in ctx.tracks:
                if self._near(zone, t, ctx):
                    st.last_person[t.track_id] = (now, ctx.norm_bbox(t.bbox), t.conf)
            for tid in [k for k, v in st.last_person.items() if (now - v[0]).total_seconds() > self.person_window_s]:
                del st.last_person[tid]
        if self._last_check is not None and (now - self._last_check).total_seconds() < 1.0 / self.check_hz:
            return []
        self._last_check = now
        h, w = ctx.image.shape[:2]
        small_w, small_h = self.width, max(1, round(self.width * h / w))
        gray = cv2.cvtColor(ctx.image, cv2.COLOR_BGR2GRAY) if ctx.image.ndim == 3 else ctx.image
        gray = cv2.GaussianBlur(cv2.resize(gray, (small_w, small_h), interpolation=cv2.INTER_AREA), (5, 5), 0)
        gray = gray.astype(np.float32)
        out = []
        for zone in self.zones.values():
            st = self._state[zone.zone_id]
            if st.mask is None or st.mask.shape != gray.shape:
                st.mask = self._mask(zone, small_w, small_h)
                st.ring = self._ring(st.mask, self.ring_k)
            if not st.mask.any():
                continue
            people = self._people_mask(ctx, st.mask.shape)
            occluded = (people & st.mask).sum() >= self.occlude * st.mask.sum()
            d = self._check(zone, st, gray, occluded, people, now, ctx)
            if d:
                out.append(d)
        return out

    # ---------- геометрия ----------

    @staticmethod
    def _mask(zone: Zone, w: int, h: int) -> np.ndarray:
        m = np.zeros((h, w), np.uint8)
        cv2.fillPoly(m, [np.array([(x * w, y * h) for x, y in zone.polygon], np.int32)], 1)
        return m.astype(bool)

    @staticmethod
    def _ring(mask: np.ndarray, k: float) -> np.ndarray:
        ys, xs = np.nonzero(mask)
        size = max(int(k * max(xs.max() - xs.min(), ys.max() - ys.min())), 3)
        grown = cv2.dilate(mask.astype(np.uint8), np.ones((size, size), np.uint8))
        return grown.astype(bool) & ~mask

    @staticmethod
    def _people_mask(ctx: RuleContext, shape) -> np.ndarray:
        h, w = shape
        m = np.zeros(shape, bool)
        for t in ctx.tracks:
            x1, y1, x2, y2 = ctx.norm_bbox(t.bbox)
            m[int(y1 * h):int(np.ceil(y2 * h)), int(x1 * w):int(np.ceil(x2 * w))] = True
        return m

    def _near(self, zone: Zone, t, ctx: RuleContext) -> bool:
        x1, y1, x2, y2 = ctx.norm_bbox(t.bbox)
        xs, ys = [p[0] for p in zone.polygon], [p[1] for p in zone.polygon]
        zx1, zy1, zx2, zy2 = min(xs), min(ys), max(xs), max(ys)
        gap = self.near_k * (y2 - y1)                          # «рядом» — в пределах near_k роста
        return not (x2 < zx1 - gap or x1 > zx2 + gap or y2 < zy1 - gap or y1 > zy2 + gap)

    # ---------- сравнение с эталоном ----------

    def _check(self, zone: Zone, st: _ZoneState, gray: np.ndarray, occluded: bool, people: np.ndarray,
               now: datetime, ctx: RuleContext) -> Detection | None:
        if st.ref is None:                                     # учим эталон (только незаслонённые кадры)
            if not occluded:
                st.learn_start = st.learn_start or now
                st.learn.append(gray)
                if (now - st.learn_start).total_seconds() >= self.learn_s and len(st.learn) >= 3:
                    self._set_ref(st, np.median(np.stack(st.learn), axis=0))
                    st.learn.clear()
            return None
        if occluded:                                           # заслонили — ждём, отсчёт не сбрасываем
            return None
        zone_px, ring_px = st.mask & ~people, st.ring & ~people
        changed = np.abs(_norm(gray, st.mask) - st.ref) > self.diff_k
        st.ratio = float(changed[zone_px].mean()) if zone_px.any() else 0.0
        ring_changed = np.abs(_norm(gray, st.ring) - st.ring_ref) > self.diff_k
        ring_ratio = float(ring_changed[ring_px].mean()) if ring_px.sum() > 20 else 0.0
        st.excess = st.ratio - ring_ratio                      # сдвинулась вся сцена — разница маленькая
        if st.ratio < self.min_ratio or st.excess < self.min_ratio:
            st.change_since = None
            if st.alerted_at is None and not ctx.tracks:       # спокойно и пусто — подстройка под свет
                self._set_ref(st, (1 - self.adapt) * st.ref_raw + self.adapt * gray)
            if st.alerted_at is not None and (now - st.alerted_at).total_seconds() >= self.relearn_s:
                st.alerted_at = None                           # вернули как было
            return None
        st.change_since = st.change_since or now
        if st.alerted_at is not None:                          # уже сообщили — через relearn_s новый вид = эталон
            if (now - st.alerted_at).total_seconds() >= self.relearn_s:
                self._set_ref(st, gray)
                st.alerted_at, st.change_since = None, None
            return None
        held = (now - st.change_since).total_seconds()
        if held < self.hold_s:
            return None
        before = {tid: v for tid, v in st.last_person.items()
                  if (st.change_since - v[0]).total_seconds() <= self.person_window_s}
        if not before:                                         # рядом никого не было — не вандализм
            return None
        st.alerted_at = now
        last = max(v[0] for v in before.values())
        people = sorted(before.items(), key=lambda kv: kv[1][0], reverse=True)[:5]
        return Detection(
            incident_type=self.incident_type, event_type="detection", camera_id=self.camera_id, occurred_at=now,
            confidence=round(float(min(1.0, st.ratio / (2 * self.min_ratio))), 3),
            participants=[Participant(tid, tuple(round(x, 4) for x in bb), round(conf, 3))
                          for tid, (_, bb, conf) in people],
            zone_id=zone.zone_id, zone_version=zone.version,
            extra={"change_ratio": round(st.ratio, 3),
                   "seconds_since_person": round(max(0.0, (st.change_since - last).total_seconds()), 1)})

    def _set_ref(self, st: _ZoneState, gray: np.ndarray) -> None:
        st.ref_raw = gray.astype(np.float32)
        st.ref = _norm(st.ref_raw, st.mask)
        st.ring_ref = _norm(st.ref_raw, st.ring)
