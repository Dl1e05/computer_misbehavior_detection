"""Правила-плагины: общий контракт.

Новое правило = файл в rules/ с @register("имя") + строка в cameras.yaml.
Правило получает RuleContext на каждый кадр и возвращает список Detection.
Detection -> Event (event_id, model_version, run_id) делает Publisher (шаг 8).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

RULES: dict[str, type["Rule"]] = {}


def register(name):
    """Декоратор: @register("fall") кладёт класс в RULES под именем "fall"."""
    def deco(cls: type["Rule"]) -> type["Rule"]:
        if name in RULES:
            raise ValueError(f"Правило '{name}' уже зарегистрировано")
        RULES[name] = cls
        return cls
    return deco


@dataclass
class Participant:
    track_id: int
    bbox: tuple[float, float, float, float]   # доли 0–1 от ширины/высоты кадра
    confidence: float


@dataclass
class Detection:
    """Что нашло правило. Поля совпадают с Event из шага 8."""
    incident_type: str                         # zone_intrusion / fall / fight / fire
    event_type: str                            # detection / suspicion / confirmation
    camera_id: str
    occurred_at: datetime                      # UTC
    confidence: float
    participants: list[Participant] = field(default_factory=list)
    zone_id: str | None = None
    zone_version: int | None = None
    episode_id: str | None = None


@dataclass
class RuleContext:
    camera_id: str
    ts: datetime                               # время кадра, UTC
    frame_w: int
    frame_h: int
    tracks: list                               # list[analytics.tracker.Track], bbox в пикселях
    session: int = 0
    poses: Any = None                          # {track_id: (17, 3)} — если правилу нужны poses
    objects: Any = None
    image: Any = None                          # BGR-кадр — для правил/моделей, которым нужна картинка

    def norm_bbox(self, bbox) -> tuple[float, float, float, float]:
        x1, y1, x2, y2 = bbox
        w, h = self.frame_w, self.frame_h
        return (min(max(x1 / w, 0.0), 1.0), min(max(y1 / h, 0.0), 1.0),
                min(max(x2 / w, 0.0), 1.0), min(max(y2 / h, 0.0), 1.0))


class Rule:
    """Базовый класс правила. Каждое правило наследуется от него."""
    incident_type: str = ""
    requires: set[str] = {"persons"}

    def __init__(self, camera_id: str, params: dict[str, Any]):
        self.camera_id = camera_id
        self.params = params

    def update(self, ctx: RuleContext) -> list[Detection]:
        """Вызывается на каждый кадр. Возвращает найденные события (обычно пусто)."""
        raise NotImplementedError

    def reset(self) -> None:
        """Камера переподключилась: старые track_id недействительны, состояние сбросить."""
