"""Слежение за предметами (рюкзак, сумка, чемодан, ноутбук) и их «владельцами» — общая логика
для правил abandoned_object и object_taken (шаг 11, пакет B).

Для каждого предмета (трек детектора предметов):
- неподвижен — центр сместился меньше static_move его размера за static_s; иначе его несут;
- «рядом» — человек ближе near_k своих ростов от центра предмета (0, если предмет внутри рамки);
- ВЛАДЕЛЕЦ — кто был рядом, когда предмет поставили (ближайший); запоминаем его внешний вид
  (цветовую гистограмму одежды — не лицо и не биометрия), чтобы узнать, если трекер сменил ему ID;
- кто-то стоит рядом attend_s подряд, а владельца нет — он становится «присматривающим» владельцем
  (сменился ID владельца, сумка у ног сидящего, друг посторожил);
- ПРИНЕСЁН: предмет видели в движении вместе с человеком (его несли). Предметы, которые с самого
  начала стоят в кадре (ноутбуки на столах, стулья, которые детектор принял за сумку), — часть сцены:
  без этого любая офисная камера через abandon_s после старта давала бы тревоги (замер на RWF-2000);
- ОСТАВЛЕН: принесённый предмет неподвижен и без присмотра abandon_s -> abandoned_object (одно на эпизод);
- ЗАБРАЛ ДРУГОЙ: принесённый и поставленный предмет с известным владельцем поехал вместе с человеком,
  который не владелец (и не похож
  на него), а владельца рядом нет -> событие object_taken. Это сигнал на проверку, не доказательство
  кражи (решение BE). Если предмет просто исчез рядом с чужим, ждём confirm_s: снова появился на
  месте (его заслоняли) — отбой; нет, а человек ушёл — object_taken;
- пропавший неподвижный предмет помним ghost_s: появился снова примерно там же — продолжает историю.
Ограничения: мелкие предметы YOLO26n находит плохо; владелец определяется по треку — при плотной
толпе и долгих перекрытиях возможны ошибки (поэтому — сигнал на проверку).
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime

import cv2
import numpy as np

from vision.rules.base import Detection, Participant, RuleContext

ABANDONED, TAKEN = "abandoned_object", "object_taken"


def _center(b) -> tuple[float, float]:
    return ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)


def _size(b) -> float:
    return max(b[2] - b[0], b[3] - b[1], 1.0)


def _rect_dist(p, b) -> float:
    """Расстояние от точки до прямоугольника (0 — внутри)."""
    dx = max(b[0] - p[0], 0.0, p[0] - b[2])
    dy = max(b[1] - p[1], 0.0, p[1] - b[3])
    return float(np.hypot(dx, dy))


def _iou(a, b) -> float:
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def appearance(image, bbox) -> np.ndarray | None:
    """Цвет одежды: HSV-гистограмма рамки человека (без верхних 20% — там голова)."""
    if image is None:
        return None
    h, w = image.shape[:2]
    x1, y1, x2, y2 = (int(max(0, bbox[0])), int(max(0, bbox[1])), int(min(w, bbox[2])), int(min(h, bbox[3])))
    y1 = y1 + (y2 - y1) // 5
    if x2 - x1 < 4 or y2 - y1 < 4:
        return None
    hsv = cv2.cvtColor(image[y1:y2, x1:x2], cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [16, 8], [0, 180, 0, 256])
    return cv2.normalize(hist, hist).flatten()


@dataclass
class _Obj:
    obj_id: int
    cls_name: str
    bbox: tuple
    conf: float
    first_seen: datetime
    last_seen: datetime
    hist: deque = field(default_factory=deque)            # (ts, cx, cy, size)
    static_since: datetime | None = None
    placed: bool = False                                  # стоял неподвижно с владельцем/без — «поставлен»
    brought: bool = False                                 # видели, как его несли (иначе — часть сцены)
    owner: int | None = None
    owner_look: np.ndarray | None = None
    owner_near_at: datetime | None = None
    near_since: dict = field(default_factory=dict)        # person_id -> с какого момента рядом
    unattended_since: datetime | None = None
    alerted_abandoned: bool = False
    alerted_taken: bool = False
    vanish_suspect: tuple | None = None                   # (person_id, с какого момента)
    last_near: dict = field(default_factory=dict)         # person_id -> когда последний раз рядом


class ObjectWatcher:
    def __init__(self, camera_id: str, params: dict):
        p = params
        self.camera_id = camera_id
        self.classes = set(p.get("classes", ["backpack", "handbag", "suitcase", "laptop"]))
        self.min_conf = float(p.get("min_conf", 0.25))
        self.abandon_s = float(p.get("abandon_s", 30.0))
        self.attend_s = float(p.get("attend_s", 5.0))
        self.near_k = float(p.get("near_k", 1.0))
        self.static_s = float(p.get("static_s", 2.0))
        self.static_move = float(p.get("static_move", 0.3))
        self.confirm_s = float(p.get("confirm_s", 2.0))
        self.ghost_s = float(p.get("ghost_s", 30.0))
        self.same_look = float(p.get("same_look", 0.35))     # расстояние Бхаттачарьи: меньше — тот же человек
        self.objs: dict[int, _Obj] = {}
        self.ghosts: list[_Obj] = []
        self._next_virtual = 1_000_000

    def reset(self) -> None:
        self.objs.clear()
        self.ghosts.clear()

    # ---------- вспомогательное ----------

    def _near(self, o: _Obj, persons) -> list:
        c = _center(o.bbox)
        out = []
        for t in persons:
            height = max(t.bbox[3] - t.bbox[1], 1.0)
            if _rect_dist(c, t.bbox) <= self.near_k * height:
                out.append(t)
        return out

    def _same_person(self, o: _Obj, t, image) -> bool:
        if o.owner is not None and t.track_id == o.owner:
            return True
        if o.owner_look is None or image is None:
            return False
        look = appearance(image, t.bbox)
        return look is not None and cv2.compareHist(o.owner_look, look, cv2.HISTCMP_BHATTACHARYYA) < self.same_look

    def _set_owner(self, o: _Obj, t, image) -> None:
        o.owner = t.track_id
        look = appearance(image, t.bbox)
        if look is not None:
            o.owner_look = look

    def _detection(self, kind: str, o: _Obj, ctx: RuleContext, people, conf: float, details: dict) -> Detection:
        return Detection(
            incident_type=kind, event_type="detection", camera_id=self.camera_id, occurred_at=ctx.ts,
            confidence=round(float(np.clip(conf, 0.0, 1.0)), 3),
            participants=[Participant(t.track_id, ctx.norm_bbox(t.bbox), round(t.conf, 3)) for t in people],
            bbox=tuple(round(v, 4) for v in ctx.norm_bbox(o.bbox)), extra={"object_class": o.cls_name, **details})

    # ---------- основной цикл ----------

    def update(self, ctx: RuleContext) -> list[Detection]:
        now = ctx.ts
        names = ctx.object_names or {}
        persons = list(ctx.tracks)
        by_id = {t.track_id: t for t in persons}
        out: list[Detection] = []
        seen = set()
        for t in ctx.objects or []:
            name = names.get(int(t.cls), str(int(t.cls)))
            if name not in self.classes or t.conf < self.min_conf:
                continue
            o = self.objs.get(t.track_id) or self._inherit(t, name, now)
            o.bbox, o.conf, o.last_seen = t.bbox, t.conf, now
            seen.add(o.obj_id)
            out += self._step(o, ctx, persons, by_id)
        # пропавшие предметы: проверить «унесли?», неподвижные — в призраки
        for oid in [k for k in self.objs if k not in seen]:
            o = self.objs[oid]
            out += self._vanished(o, ctx, persons, by_id)
        self.ghosts = [g for g in self.ghosts if (now - g.last_seen).total_seconds() <= self.ghost_s]
        return out

    def _inherit(self, t, name: str, now: datetime) -> _Obj:
        """Новый трек предмета: если там же недавно пропал такой же — это он (заслоняли)."""
        best = max((g for g in self.ghosts if g.cls_name == name), key=lambda g: _iou(g.bbox, t.bbox), default=None)
        if best is not None and _iou(best.bbox, t.bbox) >= 0.3:
            self.ghosts.remove(best)
            best.obj_id, best.vanish_suspect = t.track_id, None
            self.objs[t.track_id] = best
            return best
        o = _Obj(t.track_id, name, t.bbox, t.conf, now, now)
        self.objs[t.track_id] = o
        return o

    def _step(self, o: _Obj, ctx: RuleContext, persons, by_id) -> list[Detection]:
        now = ctx.ts
        o.vanish_suspect = None
        c, size = _center(o.bbox), _size(o.bbox)
        o.hist.append((now, c[0], c[1], size))
        while o.hist and (now - o.hist[0][0]).total_seconds() > self.static_s:
            o.hist.popleft()
        span = (now - o.hist[0][0]).total_seconds()
        moved = float(np.hypot(c[0] - o.hist[0][1], c[1] - o.hist[0][2]))
        moving = moved > self.static_move * size
        near = self._near(o, persons)
        near_ids = {t.track_id for t in near}
        for pid in list(o.near_since):
            if pid not in near_ids:
                del o.near_since[pid]
        for t in near:
            o.near_since.setdefault(t.track_id, now)
            o.last_near[t.track_id] = now
        owner_near = o.owner is not None and o.owner in near_ids
        if owner_near:
            o.owner_near_at = now
        out: list[Detection] = []

        if moving:
            carrier = min(near, key=lambda t: _rect_dist(c, t.bbox), default=None)
            if (o.placed and o.brought and o.owner is not None and carrier is not None and not o.alerted_taken
                    and not owner_near and not self._same_person(o, carrier, ctx.image)):
                o.alerted_taken = True
                out.append(self._detection(TAKEN, o, ctx, [carrier], min(o.conf, carrier.conf), {}))
            # несут — новый эпизод; несущий — новый «владелец»
            o.static_since, o.placed, o.unattended_since, o.alerted_abandoned = None, False, None, False
            if carrier is not None:
                o.brought = True                                # его несут — значит, принесён, а не часть сцены
                if o.owner is None or not owner_near:
                    self._set_owner(o, carrier, ctx.image)      # кто несёт — тот теперь и «владелец»
            return out

        if span < 0.6 * self.static_s:                       # ещё не знаем, стоит ли
            return out
        o.static_since = o.static_since or now
        if not o.placed:
            o.placed, o.alerted_taken = True, False
            if near:                                           # кто рядом при постановке — владелец
                self._set_owner(o, min(near, key=lambda t: _rect_dist(c, t.bbox)), ctx.image)
            if o.owner is not None and o.owner in near_ids:
                o.owner_near_at = now
        attended = owner_near
        if not attended:                                       # кто-то давно рядом — присматривает
            for t in near:
                if (now - o.near_since[t.track_id]).total_seconds() >= self.attend_s \
                        or self._same_person(o, t, ctx.image):
                    self._set_owner(o, t, ctx.image)
                    attended = True
                    break
        if attended:
            o.unattended_since = None
            return out
        o.unattended_since = o.unattended_since or now
        left_for = (now - o.unattended_since).total_seconds()
        if left_for >= self.abandon_s and not o.alerted_abandoned and o.brought:
            o.alerted_abandoned = True
            owner = by_id.get(o.owner) if o.owner is not None else None
            out.append(self._detection(ABANDONED, o, ctx, [owner] if owner else [], o.conf,
                                       {"unattended_s": round(left_for, 1)}))
        return out

    def _vanished(self, o: _Obj, ctx: RuleContext, persons, by_id) -> list[Detection]:
        now = ctx.ts
        gone = (now - o.last_seen).total_seconds()
        out: list[Detection] = []
        if o.placed and o.brought and o.owner is not None and not o.alerted_taken:
            recent = [pid for pid, ts in o.last_near.items() if (o.last_seen - ts).total_seconds() <= 1.0]
            owner_recent = o.owner is not None and o.owner in recent
            if o.vanish_suspect is None:
                others = [by_id[p] for p in recent if p in by_id and p != o.owner]
                if others and not owner_recent:
                    c = _center(o.bbox)
                    o.vanish_suspect = (min(others, key=lambda t: _rect_dist(c, t.bbox)).track_id, now)
            elif gone >= self.confirm_s:
                pid = o.vanish_suspect[0]
                t = by_id.get(pid)
                if t is not None and t not in self._near(o, persons) and not self._same_person(o, t, ctx.image):
                    o.alerted_taken = True                     # исчез вместе с чужим, и тот ушёл
                    out.append(self._detection(TAKEN, o, ctx, [t], min(o.conf, t.conf), {}))
                o.vanish_suspect = None
        if gone > self.confirm_s + 0.5 and o.vanish_suspect is None:   # решение принято — в призраки
            del self.objs[o.obj_id]
            if o.placed and not o.alerted_taken:
                self.ghosts.append(o)
        return out
