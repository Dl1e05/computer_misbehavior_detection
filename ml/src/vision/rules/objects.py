"""Оставленный предмет и «забрал другой» (ТЗ 18, шаг 11, пакет B).

Логика — analytics/objects.py (ObjectWatcher): предметы, владельцы, присмотр, исчезновение.
Два правила — два типа событий (каждое можно включить на камере отдельно):
- abandoned_object: предмет стоит без присмотра abandon_s; участник — владелец, если он ещё в кадре
  (может быть пусто: владелец ушёл); bbox — рамка предмета; details: object_class, unattended_s.
- object_taken: поставленный предмет унёс не владелец, пока владельца рядом не было; участник — кто унёс;
  bbox — рамка предмета; details: object_class. Инцидент на проверку, не доказательство кражи.
Лиц нет ни для одного из них (решение BE 25.11): эти типы не в FACE_EVENTS.
Детектор ищет предметы только на камерах, где включено одно из этих правил (requires: objects).
"""
from __future__ import annotations

from vision.analytics.objects import ABANDONED, TAKEN, ObjectWatcher
from vision.rules.base import Detection, Rule, RuleContext, register


class _ObjectRule(Rule):
    requires = {"persons", "objects"}

    def __init__(self, camera_id: str, params: dict):
        super().__init__(camera_id, params)
        self.watcher = ObjectWatcher(camera_id, params)

    def reset(self) -> None:
        self.watcher.reset()

    def update(self, ctx: RuleContext) -> list[Detection]:
        return [d for d in self.watcher.update(ctx) if d.incident_type == self.incident_type]

    def state_of(self, obj_id: int) -> str | None:
        """Для отрисовки: что с предметом сейчас."""
        o = self.watcher.objs.get(obj_id)
        if o is None:
            return None
        if o.alerted_abandoned:
            return "ABANDONED"
        if o.unattended_since is not None:
            return "unattended"
        return "owner" if o.owner is not None else ("static" if o.placed else "moving")


@register("abandoned_object")
class AbandonedObjectRule(_ObjectRule):
    incident_type = ABANDONED


@register("object_taken")
class ObjectTakenRule(_ObjectRule):
    incident_type = TAKEN
