"""Зоны и линии камер.

Зона  = {zone_id, camera_id, polygon, version, active, schedule?}
Линия = {line_id, camera_id, points: [p1, p2], version, active}
Координаты — доли 0–1 от ширины/высоты кадра, начало в левом верхнем углу (так же отдаёт
API BE), поэтому смена разрешения потока геометрию не ломает.

Расписание зоны (необязательно): когда зона «охраняется». Нет расписания — всегда.
    "schedule": [{"days": "mon-fri", "from": "22:00", "to": "07:00"},   # ночь, через полночь
                 {"days": "sat,sun", "from": "00:00", "to": "24:00"}]   # выходные целиком
Время — местное: "timezone" в начале zones.json (например "Asia/Almaty"), иначе часовой пояс машины.
Ночное окно (from > to) относится к дню, в который оно НАЧАЛОСЬ: пятница 22:00 → суббота 07:00.

Сейчас всё читается из configs/zones.json (рисует scripts/draw_zone.py),
потом — из GET /api/v1/ml/cameras-zones (sources/registry.py).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, time, timedelta, tzinfo
from pathlib import Path
from zoneinfo import ZoneInfo

from shapely.geometry import Point, Polygon
from shapely.prepared import prep

DAYS = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]


def _parse_days(spec) -> set[int]:
    """'mon-fri' / 'sat,sun' / ['mon', 'wed'] / 'daily' -> {0..6}."""
    if spec in (None, "", "daily", "all", "*"):
        return set(range(7))
    parts = spec if isinstance(spec, list) else str(spec).replace(" ", "").split(",")
    out: set[int] = set()
    for part in parts:
        part = str(part).lower()
        if "-" in part:
            a, b = (DAYS.index(x[:3]) for x in part.split("-"))
            out |= set(range(a, b + 1)) if a <= b else set(range(a, 7)) | set(range(0, b + 1))
        else:
            out.add(DAYS.index(part[:3]))
    return out


def _parse_time(s: str) -> time:
    if s in ("24:00", "24:00:00"):
        return time.max
    h, m = (int(x) for x in s.split(":")[:2])
    return time(h, m)


@dataclass
class Window:
    days: set[int]
    start: time
    end: time

    @classmethod
    def from_dict(cls, d: dict) -> "Window":
        return cls(_parse_days(d.get("days")), _parse_time(d.get("from", "00:00")), _parse_time(d.get("to", "24:00")))

    def contains(self, local: datetime) -> bool:
        t, wd = local.time(), local.weekday()
        if self.start <= self.end:                                  # обычное окно внутри суток
            if wd not in self.days:
                return False
            return t >= self.start if self.end == time.max else self.start <= t < self.end
        # через полночь: вечер дня из days или утро следующего за ним дня
        return (wd in self.days and t >= self.start) or ((wd - 1) % 7 in self.days and t < self.end)

    def to_dict(self) -> dict:
        return {"days": ",".join(DAYS[d] for d in sorted(self.days)),
                "from": self.start.strftime("%H:%M"), "to": "24:00" if self.end == time.max else self.end.strftime("%H:%M")}


def local_tz(name: str | None) -> tzinfo:
    if name:
        return ZoneInfo(name)
    return datetime.now().astimezone().tzinfo


@dataclass
class Zone:
    zone_id: str
    camera_id: str
    polygon: list[tuple[float, float]]
    version: int = 1
    active: bool = True
    schedule: list[dict] | None = None
    tz: str | None = None                       # из "timezone" файла; None — пояс машины
    _shape: object = field(default=None, init=False, repr=False, compare=False)
    _windows: list[Window] = field(default_factory=list, init=False, repr=False, compare=False)

    def __post_init__(self):
        self.polygon = [(float(x), float(y)) for x, y in self.polygon]
        if len(self.polygon) < 3:
            raise ValueError(f"зона {self.zone_id}: нужно минимум 3 точки")
        if not all(0.0 <= v <= 1.0 for p in self.polygon for v in p):
            raise ValueError(f"зона {self.zone_id}: координаты должны быть в долях 0–1")
        if not Polygon(self.polygon).is_valid:
            raise ValueError(f"зона {self.zone_id}: самопересекающийся полигон")
        try:
            self._windows = [Window.from_dict(w) for w in (self.schedule or [])]
        except (ValueError, KeyError) as e:
            raise ValueError(f"зона {self.zone_id}: неверное расписание {self.schedule}: {e}") from e

    def contains(self, x: float, y: float) -> bool:
        """Точка в долях 0–1. Граница считается внутри."""
        if self._shape is None:                       # строим один раз, при первой проверке
            self._shape = prep(Polygon(self.polygon))
        return self._shape.covers(Point(x, y))

    def armed(self, ts: datetime) -> bool:
        """Охраняется ли зона в момент ts (UTC). Без расписания — всегда."""
        if not self._windows:
            return True
        local = ts.astimezone(local_tz(self.tz))
        return any(w.contains(local) for w in self._windows)

    def __getstate__(self):
        # подготовленный полигон shapely не сериализуется (нужно для процессов в v2)
        state = self.__dict__.copy()
        state["_shape"] = None
        return state

    def to_dict(self) -> dict:
        d = {"zone_id": self.zone_id, "camera_id": self.camera_id,
             "polygon": [list(p) for p in self.polygon],
             "version": self.version, "active": self.active}
        if self.schedule:
            d["schedule"] = self.schedule
        return d


@dataclass
class Line:
    """Отрезок p1 -> p2. «Слева/справа» — если смотреть вдоль линии от p1 к p2 (как на экране).
    in  — переход слева направо, out — справа налево."""
    line_id: str
    camera_id: str
    points: list[tuple[float, float]]
    version: int = 1
    active: bool = True

    def __post_init__(self):
        self.points = [(float(x), float(y)) for x, y in self.points]
        if len(self.points) != 2:
            raise ValueError(f"линия {self.line_id}: нужно ровно 2 точки")
        if not all(0.0 <= v <= 1.0 for p in self.points for v in p):
            raise ValueError(f"линия {self.line_id}: координаты должны быть в долях 0–1")
        if self.points[0] == self.points[1]:
            raise ValueError(f"линия {self.line_id}: точки совпадают")

    def side(self, x: float, y: float) -> float:
        """> 0 — справа, < 0 — слева (ось y вниз, как в кадре)."""
        (x1, y1), (x2, y2) = self.points
        return (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1)

    def distance(self, x: float, y: float) -> float:
        """Расстояние со знаком (доли кадра): > 0 — справа, < 0 — слева."""
        (x1, y1), (x2, y2) = self.points
        return self.side(x, y) / ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5

    def along(self, x: float, y: float) -> float:
        """Проекция точки на линию: 0 — p1, 1 — p2 (за пределами — вне отрезка)."""
        (x1, y1), (x2, y2) = self.points
        dx, dy = x2 - x1, y2 - y1
        return ((x - x1) * dx + (y - y1) * dy) / (dx * dx + dy * dy)

    def to_dict(self) -> dict:
        return {"line_id": self.line_id, "camera_id": self.camera_id, "points": [list(p) for p in self.points],
                "version": self.version, "active": self.active}


def foot_point(bbox, frame_w: int, frame_h: int) -> tuple[float, float]:
    """Где человек стоит: середина нижней границы рамки, в долях кадра."""
    x1, _, x2, y2 = bbox
    return ((x1 + x2) / 2 / frame_w, y2 / frame_h)


def load_zones(path: str | Path) -> dict[str, Zone | Line]:
    """zones.json -> {id: Zone | Line} (зоны и линии вместе, id уникальны). Нет файла — пусто."""
    path = Path(path)
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    tz = data.get("timezone")
    out: dict[str, Zone | Line] = {}
    for z in data.get("zones", []):
        out[z["zone_id"]] = Zone(**{**z, "tz": z.get("tz", tz)})
    for ln in data.get("lines", []):
        if ln["line_id"] in out:
            raise ValueError(f"id {ln['line_id']} занят зоной")
        out[ln["line_id"]] = Line(**ln)
    return out


def save_zones(path: str | Path, items: dict[str, Zone | Line], timezone: str | None = None) -> None:
    path = Path(path)
    old = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    data: dict = {}
    if timezone or old.get("timezone"):
        data["timezone"] = timezone or old["timezone"]
    data["zones"] = [z.to_dict() for z in items.values() if isinstance(z, Zone)]
    lines = [ln.to_dict() for ln in items.values() if isinstance(ln, Line)]
    if lines:
        data["lines"] = lines
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def zones_of(items: dict, camera_id: str) -> list[Zone]:
    return [z for z in items.values() if isinstance(z, Zone) and z.camera_id == camera_id]


def lines_of(items: dict, camera_id: str) -> list[Line]:
    return [ln for ln in items.values() if isinstance(ln, Line) and ln.camera_id == camera_id]
