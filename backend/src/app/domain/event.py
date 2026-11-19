from enum import StrEnum

from app.domain.incident import IncidentType


class EventType(StrEnum):
    """Stage of an ML event (SRS §7.2; mapping to transitions is OQ-19)."""

    SUSPICION = "suspicion"
    CONFIRMATION = "confirmation"
    DETECTION = "detection"


class FailureCode(StrEnum):
    """Why an event was rejected (SRS §7.4), in check order V1..V17."""

    MALFORMED_JSON = "MALFORMED_JSON"
    MISSING_FIELD = "MISSING_FIELD"
    UNSUPPORTED_SCHEMA = "UNSUPPORTED_SCHEMA"
    INVALID_EVENT_ID = "INVALID_EVENT_ID"
    INVALID_EVENT_TYPE = "INVALID_EVENT_TYPE"
    INVALID_INCIDENT_TYPE = "INVALID_INCIDENT_TYPE"
    INVALID_STAGE_TYPE_PAIR = "INVALID_STAGE_TYPE_PAIR"
    UNKNOWN_CAMERA = "UNKNOWN_CAMERA"
    INVALID_TIMESTAMP = "INVALID_TIMESTAMP"
    INVALID_ZONE = "INVALID_ZONE"
    INVALID_PARTICIPANTS = "INVALID_PARTICIPANTS"
    INVALID_CONFIDENCE = "INVALID_CONFIDENCE"
    BIOMETRIC_NOT_ALLOWED = "BIOMETRIC_NOT_ALLOWED"
    INVALID_EMBEDDING = "INVALID_EMBEDDING"
    # Proposed, not in the SRS: an unusable value in a field that has no code
    # of its own (e.g. model_version too long). Ranked after V1..V15, so it
    # never hides a real rule failure.
    INVALID_FIELD = "INVALID_FIELD"
    DUPLICATE_EVENT = "DUPLICATE_EVENT"
    EVENT_ID_CONFLICT = "EVENT_ID_CONFLICT"
    INVALID_TRANSITION = "INVALID_TRANSITION"


# V7: which incident types each stage may carry (OQ-19 interpretation).
INCIDENT_TYPES_BY_EVENT_TYPE: dict[EventType, frozenset[IncidentType]] = {
    EventType.SUSPICION: frozenset({IncidentType.FIGHT}),
    EventType.CONFIRMATION: frozenset({IncidentType.FIGHT}),
    EventType.DETECTION: frozenset(
        {IncidentType.ZONE_INTRUSION, IncidentType.FALL, IncidentType.FIRE}
    ),
}

# Stages that create an Incident in NEW (a suspicion never does).
INCIDENT_CREATING_EVENT_TYPES = frozenset({EventType.DETECTION, EventType.CONFIRMATION})

# V12: fire may be detected without any person in view.
TYPES_WITHOUT_PARTICIPANTS = frozenset({IncidentType.FIRE})

# V3: major versions of the event schema this BE understands.
SUPPORTED_SCHEMA_MAJORS = frozenset({1})
