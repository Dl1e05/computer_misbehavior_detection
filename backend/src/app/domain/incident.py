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
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class Role(StrEnum):
    OPERATOR = "operator"
    HEAD_OF_SECURITY = "head_of_security"
    ADMINISTRATOR = "administrator"


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
