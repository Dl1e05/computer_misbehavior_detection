from __future__ import annotations

from typing import Any

RULES: dict[str, type["Rule"]] = {}

def register(name):
    """Декоратор: @register("fall") кладёт класс в RULES под именем "fall"."""
    def deco(cls: type["Rule"]) -> type["Rule"]:
        if name in RULES:
            raise ValueError(f"Правило '{name}' уже зарегестрировано")
        RULES[name] = cls
        return cls
    return deco

class Rule: 
    """Базовый класс правила. Каждое правило наследуется от него."""
    incident_type: str = ""
    requires: set[str] = {"persons"}
    def __init__(self, camera_id: str, params: dict[str, Any]):
        self.camera_id = camera_id
        self.params = params

    def update(self, ctx: Any) -> list[Any]:
        """Вызывается на каждый кадр. ctx: ts, frame, tracks, poses, objects.
        Возвращает список найденных событий (пока пустой)."""
        raise NotImplementedError