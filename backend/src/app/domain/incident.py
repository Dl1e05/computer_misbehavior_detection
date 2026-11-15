from enum import StrEnum


class IncidentStatus(StrEnum):
    SUSPECTED = "SUSPECTED"
    NEW = "NEW"
    IN_PROGRESS = "IN_PROGRESS"
    CONFIRMED = "CONFIRMED"
    FALSE_ALARM = "FALSE_ALARM"
    NOT_CONFIRMED = "NOT_CONFIRMED"
    CLOSED = "CLOSED"


class IncidentType(StrEnum):
    ZONE_INTRUSION = "zone_intrusion"
    FALL = "fall"
    FIGHT = "fight"
    FIRE = "fire"


class Severity(StrEnum):
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class Role(StrEnum):
    OPERATOR = "OPERATOR"
    HEAD_OF_SECURITY = "HEAD_OF_SECURITY"
    ADMINISTRATOR = "ADMINISTRATOR"


class ClipState(StrEnum):
    PENDING = "pending"
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    NOT_REQUIRED = "not_required"


class ClipRecordState(StrEnum):
    """State of the clip file itself (ТЗ §12.4).

    Not the same as ClipState: that is the incident's summary and has
    not_required instead of error.
    """

    PENDING = "pending"
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    ERROR = "error"


TERMINAL_STATUSES = frozenset(
    {IncidentStatus.FALSE_ALARM, IncidentStatus.NOT_CONFIRMED, IncidentStatus.CLOSED}
)


SEVERITY_BY_TYPE: dict[IncidentType, Severity] = {
    IncidentType.ZONE_INTRUSION: Severity.MEDIUM,
    IncidentType.FALL: Severity.HIGH,
    IncidentType.FIGHT: Severity.CRITICAL,
    IncidentType.FIRE: Severity.CRITICAL,
}


def severity_for(incident_type: IncidentType) -> Severity:
    return SEVERITY_BY_TYPE[incident_type]


SEVERITY_RANK: dict[Severity, int] = {
    Severity.CRITICAL: 0,
    Severity.HIGH: 1,
    Severity.MEDIUM: 2,
}


class TimerKind(StrEnum):
    SUSPICION_TIMEOUT = "SUSPICION_TIMEOUT"
    ESCALATION = "ESCALATION"


class TimerState(StrEnum):
    PENDING = "PENDING"
    FIRED = "FIRED"
    CANCELLED = "CANCELLED"
