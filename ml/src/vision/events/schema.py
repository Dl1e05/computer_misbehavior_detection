"""Контракт ML -> BE (шаг 8): три сообщения MQTT, schema_version 1.0.

Согласовано с BE (21.11, 24.11):
- Event: участники — только track_id, bbox, confidence (никаких данных лица);
  bbox события — только у fire и всегда у fire; run_id — всегда; лишние поля запрещены.
- Participant (лицо) — отдельный топик, одно сообщение на (event_id, track_id),
  no_usable_face — без snapshot_ref и embedding.
- Heartbeat — каждые 5 с + Last Will {"status": "offline"}.
- Время — UTC, ISO 8601 с миллисекундами и Z; bbox — доли 0–1 от левого верхнего угла.
- QoS 1, retain никогда, сообщение <= 256 КБ.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Annotated, Literal
from uuid import UUID

from pydantic import (AfterValidator, BaseModel, ConfigDict, Field, field_serializer, field_validator,
                      model_validator)

SCHEMA_VERSION = "1.0"
TOPIC_PREFIX = "videoanalytics/v1"
MAX_BYTES = 256 * 1024
NO_USABLE_FACE = "no_usable_face"

# какие event_type допустимы у какого incident_type — СОГЛАСОВАНО с BE (schema 1.0)
EVENT_TYPES: dict[str, set[str]] = {
    "zone_intrusion": {"detection"},
    "fall": {"detection"},
    "fire": {"detection"},
    "fight": {"suspicion", "confirmation"},
}
# ПРЕДЛОЖЕНО BE 24.11 (schema 1.1, шаг 11): пока BE не подтвердил — только лог, в MQTT не уходят
# (mqtt.send_proposed: true — отправлять). Подтвердили — перенести в EVENT_TYPES.
PROPOSED_EVENT_TYPES: dict[str, set[str]] = {
    "line_crossing": {"detection"},
    "loitering": {"detection"},
    "crowding": {"detection"},
    "running": {"detection"},
    "lying_motionless": {"detection"},
    "camera_tampering": {"detection"},
}
ZONE_TYPES = {"zone_intrusion", "loitering", "crowding"}           # нужны zone_id + zone_version
NO_PARTICIPANTS = {"fire", "camera_tampering"}                      # участников нет
EXTRA_FIELDS = {                                                     # поля 1.1: только у своего типа
    "line_id": {"line_crossing"}, "direction": {"line_crossing"}, "dwell_s": {"loitering"},
    "count": {"crowding"}, "speed": {"running"}, "tamper_type": {"camera_tampering"},
}
REQUIRED_EXTRA = {"line_crossing": {"line_id", "direction"}, "loitering": {"dwell_s"}, "crowding": {"count"},
                  "camera_tampering": {"tamper_type"}}


def is_agreed(incident_type: str) -> bool:
    return incident_type in EVENT_TYPES


def schema_version_for(incident_type: str) -> str:
    return SCHEMA_VERSION if is_agreed(incident_type) else "1.1"


def topic_event(camera_id: str) -> str:
    return f"{TOPIC_PREFIX}/cameras/{camera_id}/events"


def topic_participant(camera_id: str) -> str:
    return f"{TOPIC_PREFIX}/cameras/{camera_id}/participants"


def topic_heartbeat(node_id: str) -> str:
    return f"{TOPIC_PREFIX}/ml/{node_id}/heartbeat"


def _utc(v: datetime) -> datetime:
    if v.tzinfo is None:
        raise ValueError("время без часового пояса — нужно UTC")
    return v.astimezone(timezone.utc)


def iso_ms(v: datetime) -> str:
    return v.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.") + f"{v.microsecond // 1000:03d}Z"


def _bbox(v: tuple[float, float, float, float]):
    x1, y1, x2, y2 = v
    if not all(0.0 <= c <= 1.0 for c in v):
        raise ValueError("bbox — доли 0–1")
    if x2 < x1 or y2 < y1:
        raise ValueError("bbox: x2 >= x1 и y2 >= y1")
    return v


UtcTime = Annotated[datetime, AfterValidator(_utc)]
BBox = Annotated[tuple[float, float, float, float], AfterValidator(_bbox)]
Unit = Annotated[float, Field(ge=0.0, le=1.0)]
Id = Annotated[str, Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9._:\-]+$")]


class _Msg(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    def to_bytes(self) -> bytes:
        """Сериализуем ОДИН раз: эти байты кладутся в outbox и повторяются без изменений."""
        data = self.model_dump_json(exclude_none=True).encode("utf-8")
        if len(data) > MAX_BYTES:
            raise ValueError(f"сообщение {len(data)} байт > {MAX_BYTES}")
        return data


class EventParticipant(_Msg):
    track_id: Annotated[int, Field(ge=0)]
    bbox: BBox
    confidence: Unit


class Event(_Msg):
    schema_version: Literal["1.0", "1.1"] = SCHEMA_VERSION
    event_id: UUID
    event_type: Literal["detection", "suspicion", "confirmation"]
    incident_type: Annotated[str, Field(pattern=r"^[a-z][a-z0-9_]{1,63}$")]
    camera_id: Id
    occurred_at: UtcTime
    confidence: Unit
    participants: list[EventParticipant]
    model_version: Annotated[str, Field(min_length=1, max_length=128)]
    run_id: Id
    zone_id: Id | None = None
    zone_version: Annotated[int, Field(ge=0)] | None = None
    episode_id: Annotated[str, Field(min_length=1, max_length=128)] | None = None
    bbox: BBox | None = None
    # --- 1.1 (предложено BE) ---
    line_id: Id | None = None
    direction: Literal["in", "out"] | None = None
    dwell_s: Annotated[float, Field(ge=0)] | None = None
    count: Annotated[int, Field(ge=1)] | None = None
    speed: Annotated[float, Field(ge=0)] | None = None
    tamper_type: Literal["covered", "moved", "defocused", "glare"] | None = None

    @field_serializer("occurred_at")
    def _ser_time(self, v: datetime) -> str:
        return iso_ms(v)

    @field_serializer("event_id")
    def _ser_id(self, v: UUID) -> str:
        return str(v)

    @model_validator(mode="after")
    def _rules(self):
        inc = self.incident_type
        allowed = EVENT_TYPES.get(inc) or PROPOSED_EVENT_TYPES.get(inc)
        if allowed is None:
            raise ValueError(f"incident_type '{inc}' не согласован с BE: {sorted(EVENT_TYPES)} "
                             f"(предложены: {sorted(PROPOSED_EVENT_TYPES)})")
        if self.event_type not in allowed:
            raise ValueError(f"{inc}: event_type '{self.event_type}' недопустим, можно {sorted(allowed)}")
        if not is_agreed(inc) and self.schema_version != "1.1":
            raise ValueError(f"{inc} — тип schema 1.1, а schema_version {self.schema_version}")
        for name, owners in EXTRA_FIELDS.items():
            if getattr(self, name) is not None:
                if self.schema_version == "1.0":
                    raise ValueError(f"поле {name} — только в schema 1.1")
                if inc not in owners:
                    raise ValueError(f"поле {name} — только у {sorted(owners)}")
        missing = [f for f in REQUIRED_EXTRA.get(inc, ()) if getattr(self, f) is None]
        if missing:
            raise ValueError(f"{inc}: нужны {missing}")
        if inc in ZONE_TYPES and (self.zone_id is None or self.zone_version is None):
            raise ValueError(f"{inc}: нужны zone_id и zone_version")
        if inc not in ZONE_TYPES and (self.zone_id or self.zone_version is not None):
            raise ValueError(f"zone_id / zone_version — только у {sorted(ZONE_TYPES)}")
        if self.incident_type == "fight" and not self.episode_id:
            raise ValueError("fight: нужен episode_id")
        if self.incident_type != "fight" and self.episode_id:
            raise ValueError("episode_id — только у fight")
        if inc == "fire" and self.bbox is None:
            raise ValueError("fire: нужен bbox очага")
        if inc != "fire" and self.bbox is not None:
            raise ValueError("bbox события — только у fire")
        if inc in NO_PARTICIPANTS and self.participants:
            raise ValueError(f"{inc}: participants пустой")
        if inc not in NO_PARTICIPANTS and not self.participants:
            raise ValueError(f"{inc}: нужен хотя бы один участник")
        return self


class Participant(_Msg):
    """Данные лица участника подтверждённого события (отдельный топик .../participants)."""
    schema_version: Literal["1.0"] = SCHEMA_VERSION
    event_id: UUID
    camera_id: Id
    track_id: Annotated[int, Field(ge=0)]
    snapshot_ref: Annotated[str, Field(min_length=1, max_length=512)] | None = None
    snapshot_quality: Unit | Literal["no_usable_face"]
    embedding: list[float] | None = None
    embedding_model_version: Annotated[str, Field(min_length=1, max_length=128)]

    @field_serializer("event_id")
    def _ser_id(self, v: UUID) -> str:
        return str(v)

    @field_validator("embedding")
    @classmethod
    def _emb(cls, v):
        if v is None:
            return v
        if len(v) != 512:
            raise ValueError("embedding — 512 чисел")
        norm = sum(x * x for x in v) ** 0.5
        if abs(norm - 1.0) > 1e-2:
            raise ValueError(f"embedding должен быть нормирован (длина {norm:.4f})")
        return v

    @model_validator(mode="after")
    def _face(self):
        if self.snapshot_quality == NO_USABLE_FACE:
            if self.snapshot_ref is not None or self.embedding is not None:
                raise ValueError("no_usable_face — без snapshot_ref и embedding")
        elif self.snapshot_ref is None or self.embedding is None:
            raise ValueError("есть лицо — нужны snapshot_ref и embedding")
        return self


class CameraStatus(_Msg):
    camera_id: Id
    status: Literal["online", "offline"]
    rules: dict[str, str] = {}          # {"fire": "off: нет весов ...", "fall": "on"}


class Heartbeat(_Msg):
    schema_version: Literal["1.0"] = SCHEMA_VERSION
    node_id: Id
    run_id: Id
    status: Literal["online", "offline"]
    sent_at: UtcTime | None = None      # нет только у Last Will (его текст фиксируется при подключении)
    model_version: Annotated[str, Field(max_length=128)] | None = None
    cameras: list[CameraStatus] = []

    @field_serializer("sent_at")
    def _ser_time(self, v: datetime | None) -> str | None:
        return iso_ms(v) if v else None

    @model_validator(mode="after")
    def _online(self):
        if self.status == "online" and self.sent_at is None:
            raise ValueError("online heartbeat: нужен sent_at")
        return self


MESSAGES = {"event": Event, "participant": Participant, "heartbeat": Heartbeat}


def message_kind(topic: str) -> str | None:
    """По топику — какой это тип сообщения (для mqtt_listen.py и тестов)."""
    if topic.endswith("/events"):
        return "event"
    if topic.endswith("/participants"):
        return "participant"
    if topic.endswith("/heartbeat"):
        return "heartbeat"
    return None


def parse(kind: str, payload: bytes | str):
    return MESSAGES[kind].model_validate_json(payload)


def json_schemas() -> dict[str, dict]:
    return {k: m.model_json_schema() for k, m in MESSAGES.items()}


def dumps_pretty(payload: bytes) -> str:
    return json.dumps(json.loads(payload), ensure_ascii=False, indent=1)
