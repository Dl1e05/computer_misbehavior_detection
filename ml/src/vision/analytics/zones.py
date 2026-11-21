"""Зоны камер.

Зона = {zone_id, camera_id, polygon, version, active}. Координаты — доли 0–1 от
ширины/высоты кадра, начало в левом верхнем углу (так же отдаёт API BE),
поэтому смена разрешения потока зону не ломает.

Сейчас зоны читаются из configs/zones.json (рисует scripts/draw_zone.py),
потом — из GET /api/v1/ml/cameras-zones (sources/registry.py, шаг 8).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from shapely.geometry import Point, Polygon
from shapely.prepared import prep


@dataclass
class Zone:
    zone_id: str
    camera_id: str
    polygon: list[tuple[float, float]]
    version: int = 1
    active: bool = True
    _shape: object = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self):
        self.polygon = [(float(x), float(y)) for x, y in self.polygon]
        if len(self.polygon) < 3:
            raise ValueError(f"зона {self.zone_id}: нужно минимум 3 точки")
        if not all(0.0 <= v <= 1.0 for p in self.polygon for v in p):
            raise ValueError(f"зона {self.zone_id}: координаты должны быть в долях 0–1")
        if not Polygon(self.polygon).is_valid:
            raise ValueError(f"зона {self.zone_id}: самопересекающийся полигон")

    def contains(self, x: float, y: float) -> bool:
        """Точка в долях 0–1. Граница считается внутри."""
        if self._shape is None:                       # строим один раз, при первой проверке
            self._shape = prep(Polygon(self.polygon))
        return self._shape.covers(Point(x, y))

    def __getstate__(self):
        # подготовленный полигон shapely не сериализуется (нужно для процессов в v2)
        state = self.__dict__.copy()
        state["_shape"] = None
        return state

    def to_dict(self) -> dict:
        return {"zone_id": self.zone_id, "camera_id": self.camera_id,
                "polygon": [list(p) for p in self.polygon],
                "version": self.version, "active": self.active}


def foot_point(bbox, frame_w: int, frame_h: int) -> tuple[float, float]:
    """Где человек стоит: середина нижней границы рамки, в долях кадра."""
    x1, _, x2, y2 = bbox
    return ((x1 + x2) / 2 / frame_w, y2 / frame_h)


def load_zones(path: str | Path) -> dict[str, Zone]:
    """zones.json -> {zone_id: Zone}. Нет файла — нет зон."""
    path = Path(path)
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    zones = [Zone(**z) for z in data.get("zones", [])]
    return {z.zone_id: z for z in zones}


def save_zones(path: str | Path, zones: dict[str, Zone]) -> None:
    data = {"zones": [z.to_dict() for z in zones.values()]}
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
