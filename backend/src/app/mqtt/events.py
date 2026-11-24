"""The ML → BE event contract, as received over MQTT (SRS §7.2–7.4).

These models check what a message can be checked against on its own.
Checks that need the database (V8 camera, V11 zone ownership, V16
idempotency, V17 transition) belong to event processing
(app.services.ingest). V14 needs no database: ML sends face data in a message
of its own, so an event never carries any. V15 belongs to that message.

Rule violations are raised as PydanticCustomError whose error type is the
SRS failure code, so the caller can read the code straight from the error.
"""

import re
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Annotated, Any, Self

from pydantic import (
    AfterValidator,
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    ValidationInfo,
    field_validator,
    model_validator,
)
from pydantic_core import ErrorDetails, PydanticCustomError

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

# ASCII digits only: \d would also match e.g. Arabic-Indic digits.
_SCHEMA_VERSION = re.compile(r"([0-9]+)\.([0-9]+)")

# V14: face data of a participant; not allowed in any event. Only these keys of
# a participant are checked: anywhere else they are unknown fields, which are
# ignored and never saved.
FACE_FIELDS = ("embedding", "embedding_model_version", "snapshot_ref")

# Limits of the Postgres columns for zone_version and track_id: a larger number
# would fail to save, so the message would never be acknowledged and would come
# back forever.
_INT32_MAX = 2**31 - 1
_INT64_MAX = 2**63 - 1

# The database driver saves exactly this time as -infinity, which comes back
# without a timezone and cannot be compared with other times.
_MIN_TIME = datetime.min.replace(tzinfo=UTC)


@dataclass(frozen=True)
class EventLimits:
    """Values from outside the message, passed as validation context.

    MlEvent.model_validate_json(raw, context=EventLimits(...))
    """

    received_at: datetime
    max_future: timedelta

    def __post_init__(self) -> None:
        # A naive time cannot be compared with occurred_at; failing here
        # beats crashing later on every valid event.
        if self.received_at.utcoffset() is None:
            raise ValueError("received_at must have a timezone")
        if self.max_future < timedelta(0):
            raise ValueError("max_future must be >= 0")


def _fail(code: FailureCode, message: str) -> PydanticCustomError:
    return PydanticCustomError(code.value, message)


def _without_nul(value: object) -> object:
    # Postgres text and jsonb cannot store U+0000. A value that fails to save
    # would roll back the transaction, so the message would never be
    # acknowledged and would come back forever. Rejecting it here ends that.
    if isinstance(value, str) and "\x00" in value:
        raise ValueError("text must not contain U+0000")
    return value


def _limits(info: ValidationInfo) -> EventLimits:
    # Missing limits are a programming error, not a bad message: TypeError
    # is not turned into a validation error, so it is not silently skipped.
    if not isinstance(info.context, EventLimits):
        raise TypeError("validate MlEvent with context=EventLimits(...)")
    return info.context


def _ordered(box: list[float]) -> list[float]:
    # Equal is allowed: ML clips boxes to the frame and rounds them, so a person
    # at the edge can get a box without width. That must not lose an alarm.
    if not (box[0] <= box[2] and box[1] <= box[3]):
        raise ValueError("bbox must be [x1, y1, x2, y2] with x1 <= x2 and y1 <= y2")
    return box


# [x1, y1, x2, y2] as fractions of the frame (0..1), top left to bottom right.
NormalizedBox = Annotated[
    list[Annotated[float, Field(ge=0, le=1)]],
    Field(min_length=4, max_length=4),
    AfterValidator(_ordered),
]


class Participant(BaseModel):
    model_config = _CONTRACT_CONFIG

    # ByteTrack id, counted from 0; unique per camera and run_id.
    track_id: int = Field(ge=0, le=_INT64_MAX)
    # Detection or track confidence; not an identity score.
    confidence: float = Field(ge=0, le=1)
    bbox: NormalizedBox | None = None
    # FACE_FIELDS. Declared only to be refused (V14); Any, so that malformed
    # face data is reported as face data too.
    embedding: Any = None
    embedding_model_version: Any = None
    snapshot_ref: Any = None

    def has_face_data(self) -> bool:
        return any(getattr(self, field) is not None for field in FACE_FIELDS)


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
    # One run of the ML process; track ids start again in the next run.
    run_id: str = Field(min_length=1, max_length=128)
    zone_id: str | None = None
    # Version of the zone that ML used; not required, an old one is no fault.
    zone_version: int | None = Field(default=None, ge=0, le=_INT32_MAX)
    # Detection pipeline version, e.g. "fight-temporal-1.3.0/yolo26-det-2.0.1".
    model_version: str = Field(min_length=1, max_length=128)
    # Confidence of the whole event; not an identity score.
    confidence: float | None = Field(default=None, ge=0, le=1)
    # Where the fire is. Other types have boxes on their participants.
    bbox: NormalizedBox | None = None
    participants: list[Participant] = Field(default_factory=list)
    sent_at: AwareDatetime | None = None
    # Links a fight suspicion to its confirmation.
    episode_id: str | None = None

    @field_validator("*")
    @classmethod
    def _no_nul(cls, value: object) -> object:
        return _without_nul(value)

    @field_validator("schema_version")
    @classmethod
    def _supported_schema(cls, value: str) -> str:
        # V3: any minor version of a supported major version.
        match = _SCHEMA_VERSION.fullmatch(value)
        if match is None or int(match.group(1)) not in SUPPORTED_SCHEMA_MAJORS:
            # No value in the message: it goes to logs (see Rejection).
            raise _fail(FailureCode.UNSUPPORTED_SCHEMA, "unsupported schema version")
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
        if value == _MIN_TIME:
            raise _fail(FailureCode.INVALID_TIMESTAMP, "occurred_at is not a real time")
        return value

    # Pydantic runs this only once every field is valid. So if a message has
    # a field error AND a cross-field fault, only the field error's code is
    # seen, even when the cross-field rule has the lower V number. Known
    # deviation from "first failure in V order" (§7.4): every V1..V14
    # rejection is handled the same way (§7.5), only the reported code can
    # differ, and it is always a real fault of the message.
    # Within this validator, checks follow the V numbers.
    @model_validator(mode="after")
    def _cross_field_rules(self, info: ValidationInfo) -> Self:
        limits = _limits(info)
        self._check_stage_type_pair()
        self._check_not_in_future(limits)
        self._check_zone()
        self._check_participants()
        self._check_no_face_data()
        return self

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

    def _check_no_face_data(self) -> None:
        # V14: face data has a message of its own.
        if any(participant.has_face_data() for participant in self.participants):
            raise _fail(
                FailureCode.BIOMETRIC_NOT_ALLOWED, "face data is not allowed in events"
            )


@dataclass(frozen=True)
class Rejection:
    """Why a message was rejected: one SRS failure code (§7.4, §7.5).

    detail is built from the field path and the error message only, never
    from the input: the input may be an embedding, and detail goes to logs
    and dead letters, where biometric data is not allowed (SRS §14).
    """

    code: FailureCode
    detail: str


# Codes for field errors that are not SRS rule errors of their own.
_EVENT_FIELD_CODES: dict[str, FailureCode] = {
    "schema_version": FailureCode.UNSUPPORTED_SCHEMA,
    "event_id": FailureCode.INVALID_EVENT_ID,
    "event_type": FailureCode.INVALID_EVENT_TYPE,
    "incident_type": FailureCode.INVALID_INCIDENT_TYPE,
    # The closest V rule for an unusable value (SRS has no code for it):
    "camera_id": FailureCode.UNKNOWN_CAMERA,
    "occurred_at": FailureCode.INVALID_TIMESTAMP,
    "sent_at": FailureCode.INVALID_TIMESTAMP,
    "zone_id": FailureCode.INVALID_ZONE,
    "zone_version": FailureCode.INVALID_ZONE,
    "confidence": FailureCode.INVALID_CONFIDENCE,
    "participants": FailureCode.INVALID_PARTICIPANTS,
}
_PARTICIPANT_FIELD_CODES: dict[str, FailureCode] = {
    "confidence": FailureCode.INVALID_CONFIDENCE,
}
# Rank of each code in V order (FailureCode is declared in that order).
_RANK = {code: rank for rank, code in enumerate(FailureCode)}
_FAILURE_CODES = {code.value for code in FailureCode}


def _is_required_field(loc: tuple[int | str, ...]) -> bool:
    # Only these locations are fields; deeper ones are list items.
    if len(loc) == 1:
        field = MlEvent.model_fields.get(str(loc[0]))
    elif len(loc) == 3 and loc[0] == "participants":
        field = Participant.model_fields.get(str(loc[2]))
    else:
        return False
    return field is not None and field.is_required()


def _code_for(error: ErrorDetails) -> FailureCode:
    loc = error["loc"]
    if error["type"] in _FAILURE_CODES:
        # Our own rule errors carry the code as their type.
        return FailureCode(error["type"])
    if error["type"] == "json_invalid" or loc == ():
        # Not JSON, not UTF-8, or JSON that is not an object (V1).
        return FailureCode.MALFORMED_JSON
    if error["type"] == "missing" or (
        "input" in error and error["input"] is None and _is_required_field(loc)
    ):
        # V2: required fields must be present and non-null. A null in an
        # optional field or inside a list is that field's own error.
        return FailureCode.MISSING_FIELD
    if loc[0] == "participants" and len(loc) >= 3:
        return _PARTICIPANT_FIELD_CODES.get(
            str(loc[2]), FailureCode.INVALID_PARTICIPANTS
        )
    # Fields without a code of their own (e.g. model_version too long).
    return _EVENT_FIELD_CODES.get(str(loc[0]), FailureCode.INVALID_FIELD)


def parse_event(raw: bytes, limits: EventLimits) -> MlEvent | Rejection:
    """Validate one MQTT message: the event, or one failure code.

    SRS §7.4 applies the checks in order and stops at the first failure;
    Pydantic reports every error at once, so the earliest code is picked
    (see the note on MlEvent._cross_field_rules for the one exception).
    """
    # Checked up front: inside the model the check only runs for messages
    # whose fields are all valid. (A mode="before" model validator would run
    # always, but it turns JSON input into Python input, and strict mode then
    # refuses strings for datetime and UUID fields.)
    if not isinstance(limits, EventLimits):
        raise TypeError("parse_event needs EventLimits")
    try:
        return MlEvent.model_validate_json(raw, context=limits)
    except ValidationError as error:
        errors = error.errors(include_url=False)
    coded = [(_code_for(item), item) for item in errors]
    code, first = min(coded, key=lambda pair: _RANK[pair[0]])
    path = ".".join(str(part) for part in first["loc"]) or "message"
    return Rejection(code=code, detail=f"{path}: {first['msg']}")
