"""Правила-плагины. Импорт модуля правила регистрирует его в RULES."""
from __future__ import annotations

from vision.rules.base import RULES, Detection, Participant, Rule, RuleContext, register
from vision.rules import fall, zone  # noqa: F401  — регистрируют fall, zone_intrusion


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
        if name == "zone_intrusion":
            ids = params.get("zones")
            if ids is None:
                params["zones"] = [z for z in zones.values() if z.camera_id == camera_id]
            else:
                missing = [i for i in ids if i not in zones]
                if missing:
                    raise ValueError(f"камера {camera_id}: нет зон {missing} в zones.json")
                params["zones"] = [zones[i] for i in ids]
        rules.append(RULES[name](camera_id, params))
    return rules


def required_inputs(rules: list[Rule]) -> set[str]:
    """Что нужно правилам камеры: persons, poses, ... — по этому решаем, гонять ли pose-модель."""
    out: set[str] = set()
    for r in rules:
        out |= set(r.requires)
    return out


__all__ = ["RULES", "Detection", "Participant", "Rule", "RuleContext", "register", "build_rules", "required_inputs"]
