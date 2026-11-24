"""Правила-плагины. Импорт модуля правила регистрирует его в RULES."""
from __future__ import annotations

from vision.analytics.zones import Line, Zone, find
from vision.rules.base import RULES, Detection, Participant, Rule, RuleContext, register
from vision.rules import (crowd, fall, fight, fire, line, loitering, lying, objects,  # noqa: F401 (fire: fire + smoke)
                          running, scene_change, tampering, zone)                         # — регистрируют правила


def build_rules(camera_id: str, rules_cfg: dict | None, zones: dict | None = None) -> list[Rule]:
    """Правила камеры из cameras.yaml: {имя: параметры}.

    Для zone_intrusion в параметрах — id зон ("zones: [zone-1]"); без списка
    берутся все активные зоны этой камеры.
    """
    zones = zones or {}
    rules = []
    for name, params in (rules_cfg or {}).items():
        if name not in RULES:
            raise ValueError(f"камера {camera_id}: неизвестное правило '{name}' (есть: {sorted(RULES)})")
        params = dict(params or {})
        cls = RULES[name]
        if cls.uses_zones:
            params["zones"] = _resolve(camera_id, params.get("zones"), zones, Zone, "зон")
        if cls.uses_lines:
            params["lines"] = _resolve(camera_id, params.get("lines"), zones, Line, "линий")
        rules.append(cls(camera_id, params))
    return rules


def _resolve(camera_id: str, ids, items: dict, kind: type, what: str) -> list:
    """Список из cameras.yaml (UUID или имя) -> объекты; без списка — все такие объекты этой камеры."""
    if ids is None:
        return [x for x in items.values() if isinstance(x, kind) and x.camera_id == camera_id]
    found = {i: find(items, i) for i in ids}
    missing = [i for i, x in found.items() if not isinstance(x, kind)]
    if missing:
        raise ValueError(f"камера {camera_id}: нет {what} {missing} в zones.json")
    return list(found.values())


def required_inputs(rules: list[Rule]) -> set[str]:
    """Что нужно правилам камеры: persons, poses, ... — по этому решаем, гонять ли pose-модель."""
    out: set[str] = set()
    for r in rules:
        out |= set(r.requires)
    return out


__all__ = ["RULES", "Detection", "Participant", "Rule", "RuleContext", "register", "build_rules", "required_inputs"]
