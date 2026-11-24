"""Контракт ML -> BE: три сообщения MQTT (Event, Participant, Heartbeat).

schema_version 1.0 — согласовано с BE (21.11, 24.11, правки 25.11):
- Event: участники — только track_id, bbox, confidence (никаких данных лица); run_id — всегда;
  bbox события — только у fire (и smoke) и всегда у них; лишние поля запрещены;
  zone_id — UUID зоны из cameras-zones.
- Participant (лицо) — отдельный топик, одно сообщение на (event_id, track_id):
  snapshot_quality "ok" (+ snapshot_ref, embedding, quality_score) или "no_usable_face" (без них).
- Heartbeat — раз в 5 с QoS 0 + Last Will {"status": "offline"} QoS 1.
- Время — UTC, ISO 8601 с миллисекундами и Z; bbox — доли 0–1 от левого верхнего угла.
- События и участники — QoS 1; retain никогда; сообщение <= 256 КБ.
schema_version 1.1 — новые типы (шаг 11), ПРЕДЛОЖЕНЫ BE: ждут одобрения, ML их не публикует
(mqtt.send_proposed). zone_id / line_id / bbox — сверху, остальное — в details со своей схемой на тип.

Что JSON Schema выразить не может (проверяет только наш валидатор, BE — по описанию):
x1 <= x2 и y1 <= y2 в bbox; длина вектора лица ≈ 1; связи полей между собой (см. описание Event).
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Annotated, Literal, Union
from uuid import UUID

from pydantic import (AfterValidator, BaseModel, ConfigDict, Field, field_serializer, field_validator,
                      model_validator)

SCHEMA_VERSION = "1.0"
TOPIC_PREFIX = "videoanalytics/v1"
MAX_BYTES = 256 * 1024
NO_USABLE_FACE = "no_usable_face"
FACE_OK = "ok"

# какие event_type допустимы у какого incident_type — СОГЛАСОВАНО с BE (schema 1.0)
EVENT_TYPES: dict[str, set[str]] = {
    "zone_intrusion": {"detection"},
    "fall": {"detection"},
    "fire": {"detection"},
    "fight": {"suspicion", "confirmation"},
}
# ПРЕДЛОЖЕНО BE (schema 1.1): ждут одобрения руководителя BE и развёртывания BE — только лог.
# Важность (предложение BE): smoke — критично; lying_motionless, camera_tampering, abandoned_object — высокая.
PROPOSED_EVENT_TYPES: dict[str, set[str]] = {
    "line_crossing": {"detection"},
    "loitering": {"detection"},
    "crowding": {"detection"},
    "running": {"detection"},
    "lying_motionless": {"detection"},
    "camera_tampering": {"detection"},
    "smoke": {"detection"},
}
ZONE_TYPES = {"zone_intrusion", "loitering", "crowding"}           # нужны zone_id + zone_version
FIRE_LIKE = {"fire", "smoke"}                                       # bbox очага, участников нет
NO_PARTICIPANTS = FIRE_LIKE | {"camera_tampering"}


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


def _bbox(v):
    x1, y1, x2, y2 = v
    if x2 < x1 or y2 < y1:
        raise ValueError("bbox: x2 >= x1 и y2 >= y1")
    return v


def _require_version(schema: dict, *_) -> None:
    """JSON Schema: schema_version обязателен (в Python у него есть значение по умолчанию)."""
    if "schema_version" in schema.get("properties", {}):
        req = schema.setdefault("required", [])
        if "schema_version" not in req:
            req.insert(0, "schema_version")


Unit = Annotated[float, Field(ge=0.0, le=1.0)]
UtcTime = Annotated[datetime, AfterValidator(_utc),
                    Field(description="UTC, ISO 8601 с миллисекундами и Z: 2026-11-21T05:15:32.480Z")]
BBox = Annotated[tuple[Unit, Unit, Unit, Unit], AfterValidator(_bbox),
                 Field(description="[x1, y1, x2, y2] — доли 0–1 от ширины/высоты кадра, начало — левый верхний "
                                   "угол; x1 <= x2, y1 <= y2")]
Id = Annotated[str, Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9._:\-]+$")]
Embedding = Annotated[list[float], Field(min_length=512, max_length=512,
                                         description="512 чисел ArcFace, нормирован: длина вектора 1 ± 0.01")]


class _Msg(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, json_schema_extra=_require_version)

    @model_validator(mode="before")
    @classmethod
    def _need_version(cls, data, info):
        """Входящий JSON (parse) без schema_version — ошибка; в коде у поля есть значение по умолчанию."""
        if (info.context or {}).get("require_version") and isinstance(data, dict) \
                and "schema_version" in cls.model_fields and "schema_version" not in data:
            raise ValueError("schema_version обязателен")
        return data

    def to_bytes(self) -> bytes:
        """Сериализуем ОДИН раз: эти байты кладутся в outbox и повторяются без изменений."""
        data = self.model_dump_json(exclude_none=True).encode("utf-8")
        if len(data) > MAX_BYTES:
            raise ValueError(f"сообщение {len(data)} байт > {MAX_BYTES}")
        return data


class EventParticipant(_Msg):
    track_id: Annotated[int, Field(ge=0, description="целое >= 0; уникален внутри camera_id и run_id")]
    bbox: BBox
    confidence: Unit


# ---------- details (schema 1.1): своя схема на каждый новый тип ----------

class LineCrossingDetails(_Msg):
    """line_crossing: in — слева направо, если смотреть вдоль линии от первой точки ко второй."""
    direction: Literal["in", "out"]


class LoiteringDetails(_Msg):
    """loitering: сколько секунд человек в зоне на момент события."""
    dwell_s: Annotated[float, Field(ge=0)]


class CrowdingDetails(_Msg):
    """crowding: сколько людей в зоне на момент события."""
    count: Annotated[int, Field(ge=1)]


class RunningDetails(_Msg):
    """running: скорость в ростах человека в секунду."""
    speed: Annotated[float, Field(ge=0)]


class TamperingDetails(_Msg):
    """camera_tampering: что случилось с камерой."""
    tamper_type: Literal["covered", "moved", "defocused", "glare"]


DETAILS: dict[str, type[_Msg]] = {
    "line_crossing": LineCrossingDetails, "loitering": LoiteringDetails, "crowding": CrowdingDetails,
    "running": RunningDetails, "camera_tampering": TamperingDetails,
}
Details = Union[LineCrossingDetails, LoiteringDetails, CrowdingDetails, RunningDetails, TamperingDetails]


class Event(_Msg):
    """Событие ML. Связи полей (проверяет валидатор ML; BE — по этому описанию):
    - event_type: suspicion/confirmation — только у fight; у остальных — detection;
    - zone_id + zone_version — обязательны у zone_intrusion, loitering, crowding и запрещены у прочих;
    - episode_id — обязателен у fight и запрещён у прочих;
    - bbox — обязателен у fire и smoke (рамка очага) и запрещён у прочих; у них participants пустой;
    - participants не пустой у всех, кроме fire, smoke, camera_tampering;
    - schema 1.1 (новые типы): line_id — обязателен у line_crossing; details — по схеме своего типа.
    """
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
    zone_id: Annotated[UUID, Field(description="UUID зоны из cameras-zones")] | None = None
    zone_version: Annotated[int, Field(ge=0)] | None = None
    episode_id: Annotated[str, Field(min_length=1, max_length=128)] | None = None
    bbox: BBox | None = None
    # --- schema 1.1 (предложено BE) ---
    line_id: Annotated[UUID, Field(description="UUID линии из cameras-zones")] | None = None
    details: Details | None = None

    @field_serializer("occurred_at")
    def _ser_time(self, v: datetime) -> str:
        return iso_ms(v)

    @field_serializer("event_id", "zone_id", "line_id")
    def _ser_id(self, v: UUID | None) -> str | None:
        return str(v) if v is not None else None

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
        if self.schema_version == "1.0" and (self.line_id is not None or self.details is not None):
            raise ValueError("line_id и details — только в schema 1.1")
        want = DETAILS.get(inc)
        if want is None and self.details is not None:
            raise ValueError(f"{inc}: details не бывает")
        if want is not None and not isinstance(self.details, want):
            raise ValueError(f"{inc}: нужен details по схеме {want.__name__}")
        if (inc == "line_crossing") != (self.line_id is not None):
            raise ValueError("line_id — обязателен у line_crossing и только у него")
        if inc in ZONE_TYPES and (self.zone_id is None or self.zone_version is None):
            raise ValueError(f"{inc}: нужны zone_id и zone_version")
        if inc not in ZONE_TYPES and (self.zone_id is not None or self.zone_version is not None):
            raise ValueError(f"zone_id / zone_version — только у {sorted(ZONE_TYPES)}")
        if inc == "fight" and not self.episode_id:
            raise ValueError("fight: нужен episode_id")
        if inc != "fight" and self.episode_id:
            raise ValueError("episode_id — только у fight")
        if inc in FIRE_LIKE and self.bbox is None:
            raise ValueError(f"{inc}: нужен bbox очага")
        if inc not in FIRE_LIKE and self.bbox is not None:
            raise ValueError(f"bbox события — только у {sorted(FIRE_LIKE)}")
        if inc in NO_PARTICIPANTS and self.participants:
            raise ValueError(f"{inc}: participants пустой")
        if inc not in NO_PARTICIPANTS and not self.participants:
            raise ValueError(f"{inc}: нужен хотя бы один участник")
        return self


class Participant(_Msg):
    """Данные лица участника подтверждённого события (топик .../participants).
    snapshot_quality "ok" — есть snapshot_ref, embedding и quality_score; "no_usable_face" — нет ни одного из них.
    Лица — только для zone_intrusion, fall и fight/confirmation (никогда для suspicion и новых типов)."""
    schema_version: Literal["1.0"] = SCHEMA_VERSION
    event_id: UUID
    camera_id: Id
    track_id: Annotated[int, Field(ge=0)]
    snapshot_ref: Annotated[str, Field(min_length=1, max_length=512,
                                       description="ключ в хранилище: faces/<camera_id>/<ГГГГММДД UTC>/"
                                                   "<event_id>_<track_id>.jpg")] | None = None
    snapshot_quality: Literal["ok", "no_usable_face"]
    quality_score: Unit | None = None
    embedding: Embedding | None = None
    embedding_model_version: Annotated[str, Field(min_length=1, max_length=128)]

    @field_serializer("event_id")
    def _ser_id(self, v: UUID) -> str:
        return str(v)

    @field_validator("embedding")
    @classmethod
    def _emb(cls, v):
        if v is None:
            return v
        norm = sum(x * x for x in v) ** 0.5
        if abs(norm - 1.0) > 1e-2:
            raise ValueError(f"embedding должен быть нормирован (длина {norm:.4f})")
        return v

    @model_validator(mode="after")
    def _face(self):
        face = (self.snapshot_ref, self.embedding, self.quality_score)
        if self.snapshot_quality == NO_USABLE_FACE and any(x is not None for x in face):
            raise ValueError("no_usable_face — без snapshot_ref, embedding и quality_score")
        if self.snapshot_quality == FACE_OK and any(x is None for x in face):
            raise ValueError("ok — нужны snapshot_ref, embedding и quality_score")
        if self.snapshot_ref is not None:
            expected = f"_{self.track_id}.jpg"
            if not (self.snapshot_ref.endswith(f"{self.event_id}{expected}")
                    and self.snapshot_ref.startswith(f"faces/{self.camera_id}/")):
                raise ValueError("snapshot_ref не совпадает с camera_id / event_id / track_id")
        return self


class CameraStatus(_Msg):
    camera_id: Id
    status: Literal["online", "offline"]
    rules: dict[str, str] = Field(default_factory=dict,
                                  description='{"fall": "on", "fire": "off: нет весов ..."}')


class Heartbeat(_Msg):
    """Статус узла: раз в 5 с QoS 0, без retain. Last Will (QoS 1) — status offline без sent_at;
    will от старого run_id BE игнорирует."""
    schema_version: Literal["1.0"] = SCHEMA_VERSION
    node_id: Id
    run_id: Id
    status: Literal["online", "offline"]
    sent_at: UtcTime | None = None      # нет только у Last Will (его текст фиксируется при подключении)
    model_version: Annotated[str, Field(max_length=128)] | None = None
    cameras: list[CameraStatus] = Field(default_factory=list)
    dropped_stale: Annotated[int, Field(ge=0, description="сколько событий/участников с запуска не отправлено: "
                                                          "старше mqtt.max_age_s после долгой недоступности")] | None = None

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
    """Входящее сообщение строго по контракту (в т.ч. schema_version обязателен)."""
    return MESSAGES[kind].model_validate_json(payload, context={"require_version": True})


def json_schemas() -> dict[str, dict]:
    return {k: m.model_json_schema() for k, m in MESSAGES.items()}


def dumps_pretty(payload: bytes) -> str:
    return json.dumps(json.loads(payload), ensure_ascii=False, indent=1)
