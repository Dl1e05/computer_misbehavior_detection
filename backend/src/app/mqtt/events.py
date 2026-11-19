"""The ML → BE event contract, as received over MQTT (SRS §7.2–7.4).

These models check what a message can be checked against on its own.
Checks that need the database (V8 camera, V11 zone ownership, V14 repeat
events, V16 idempotency, V17 transition) belong to event processing.

Rule violations are raised as PydanticCustomError whose error type is the
SRS failure code, so the caller can read the code straight from the error.
"""

import math
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Self

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    ValidationInfo,
    field_validator,
    model_validator,
)
from pydantic_core import PydanticCustomError

from app.domain.event import (
    INCIDENT_CREATING_EVENT_TYPES,
    INCIDENT_TYPES_BY_EVENT_TYPE,
    SUPPORTED_SCHEMA_MAJORS,
    TYPES_WITHOUT_PARTICIPANTS,
    EventType,
    FailureCode,
)
from app.domain.incident import IncidentType

# extra="ignore": V3 accepts any minor version of a supported major one, so
# a field added in schema 1.1 must not make BE reject the whole event.
# strict: "0.9" is not a number and 1 is not true; JSON strings still parse
# into UUID, datetime and enum fields.
# allow_inf_nan=False: Python's json reads NaN and Infinity, and Pydantic
# would accept them as floats.
_CONTRACT_CONFIG = ConfigDict(
    extra="ignore", strict=True, frozen=True, allow_inf_nan=False
)

_SCHEMA_VERSION = re.compile(r"^(\d+)\.(\d+)$")


@dataclass(frozen=True)
class EventLimits:
    """Values from outside the message, passed as validation context.

    MlEvent.model_validate_json(raw, context=EventLimits(...))
    """

    embedding_dim: int
    received_at: datetime
    max_future: timedelta


def _fail(code: FailureCode, message: str) -> PydanticCustomError:
    return PydanticCustomError(code.value, message)


def _limits(info: ValidationInfo) -> EventLimits:
    # Missing limits are a programming error, not a bad message: TypeError
    # is not turned into a validation error, so it is not silently skipped.
    if not isinstance(info.context, EventLimits):
        raise TypeError("validate MlEvent with context=EventLimits(...)")
    return info.context


class Participant(BaseModel):
    model_config = _CONTRACT_CONFIG

    # ByteTrack id, stable within the camera session.
    track_id: str = Field(min_length=1)
    # Detection or track confidence; not an identity score.
    confidence: float = Field(ge=0, le=1)
    # [x1, y1, x2, y2]. Units (pixels or 0..1) are pending with the ML team.
    bbox: list[float] | None = Field(default=None, min_length=4, max_length=4)
    # Storage key of the face snapshot, never a URL.
    snapshot_ref: str | None = None
    # Face embedding; allowed only on some events (V14), D long (V15).
    embedding: list[float] | None = None
    embedding_model_version: str | None = None


class MlEvent(BaseModel):
    model_config = _CONTRACT_CONFIG

    schema_version: str = Field(min_length=1)
    # Reused on redelivery of the same event (idempotency key).
    event_id: uuid.UUID
    event_type: EventType
    camera_id: str = Field(min_length=1)
    # AwareDatetime refuses a time without a timezone.
    occurred_at: AwareDatetime
    incident_type: IncidentType
    zone_id: str | None = None
    # Detection pipeline version, e.g. "fight-temporal-1.3.0/yolo26-det-2.0.1".
    model_version: str = Field(min_length=1, max_length=128)
    participants: list[Participant] = Field(default_factory=list)
    sent_at: AwareDatetime | None = None
    # Links a fight suspicion to its confirmation (proposed by the ML team).
    episode_id: str | None = None

    @field_validator("schema_version")
    @classmethod
    def _supported_schema(cls, value: str) -> str:
        # V3: any minor version of a supported major version.
        match = _SCHEMA_VERSION.match(value)
        if match is None or int(match.group(1)) not in SUPPORTED_SCHEMA_MAJORS:
            raise _fail(FailureCode.UNSUPPORTED_SCHEMA, f"unsupported schema {value!r}")
        return value

    @field_validator("event_id")
    @classmethod
    def _uuid_v4(cls, value: uuid.UUID) -> uuid.UUID:
        # V4: a random (v4) UUID, not just any UUID.
        if value.version != 4:
            raise _fail(FailureCode.INVALID_EVENT_ID, "event_id must be a UUID v4")
        return value

    @field_validator("occurred_at")
    @classmethod
    def _utc(cls, value: datetime) -> datetime:
        # V9: RFC 3339 with timezone Z.
        if value.utcoffset() != timedelta(0):
            raise _fail(FailureCode.INVALID_TIMESTAMP, "occurred_at must be UTC")
        return value

    # Runs only once every field is valid, which keeps the V order: field
    # errors first. Checks follow the V numbers; the first failure is raised.
    @model_validator(mode="after")
    def _cross_field_rules(self, info: ValidationInfo) -> Self:
        limits = _limits(info)
        self._check_conditional_fields()
        self._check_stage_type_pair()
        self._check_not_in_future(limits)
        self._check_zone()
        self._check_participants()
        self._check_embeddings(limits)
        return self

    def _check_conditional_fields(self) -> None:
        # V2: embedding_model_version is required when embedding is present.
        for participant in self.participants:
            if (
                participant.embedding is not None
                and participant.embedding_model_version is None
            ):
                raise _fail(
                    FailureCode.MISSING_FIELD,
                    "embedding_model_version is required with embedding",
                )

    def _check_stage_type_pair(self) -> None:
        # V7
        if self.incident_type not in INCIDENT_TYPES_BY_EVENT_TYPE[self.event_type]:
            raise _fail(
                FailureCode.INVALID_STAGE_TYPE_PAIR,
                f"{self.event_type} cannot carry {self.incident_type}",
            )

    def _check_not_in_future(self, limits: EventLimits) -> None:
        # V9: small skew is only a warning (V10), far future is rejected.
        if self.occurred_at > limits.received_at + limits.max_future:
            raise _fail(FailureCode.INVALID_TIMESTAMP, "occurred_at is in the future")

    def _check_zone(self) -> None:
        # V11, first half; that the zone belongs to the camera needs the DB.
        if self.incident_type is IncidentType.ZONE_INTRUSION and not self.zone_id:
            raise _fail(FailureCode.INVALID_ZONE, "zone_intrusion requires zone_id")

    def _check_participants(self) -> None:
        # V12
        track_ids = [participant.track_id for participant in self.participants]
        if len(track_ids) != len(set(track_ids)):
            raise _fail(
                FailureCode.INVALID_PARTICIPANTS, "track_id repeated in the event"
            )
        if (
            not self.participants
            and self.event_type in INCIDENT_CREATING_EVENT_TYPES
            and self.incident_type not in TYPES_WITHOUT_PARTICIPANTS
        ):
            raise _fail(
                FailureCode.INVALID_PARTICIPANTS,
                f"{self.incident_type} needs at least one participant",
            )

    def _check_embeddings(self, limits: EventLimits) -> None:
        embeddings = [
            participant.embedding
            for participant in self.participants
            if participant.embedding is not None
        ]
        # V14: never on a suspicion (D-06). Repeat events need the DB.
        if embeddings and self.event_type is EventType.SUSPICION:
            raise _fail(
                FailureCode.BIOMETRIC_NOT_ALLOWED,
                "embeddings are not allowed on a suspicion",
            )
        # V15: exactly D numbers, all finite.
        for embedding in embeddings:
            if len(embedding) != limits.embedding_dim:
                raise _fail(
                    FailureCode.INVALID_EMBEDDING,
                    f"embedding must have {limits.embedding_dim} numbers",
                )
            if not all(math.isfinite(value) for value in embedding):
                raise _fail(FailureCode.INVALID_EMBEDDING, "embedding not finite")
