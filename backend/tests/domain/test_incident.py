from app.domain.incident import (
    SEVERITY_RANK,
    TERMINAL_STATUSES,
    IncidentStatus,
    IncidentType,
    Role,
    Severity,
    severity_for,
)


def test_enums_have_expected_numbers_of_members() -> None:
    assert len(IncidentStatus) == 7
    assert len(IncidentType) == 4
    assert len(Severity) == 3
    assert len(Role) == 3


def test_terminal_statuses() -> None:
    assert TERMINAL_STATUSES == {
        IncidentStatus.FALSE_ALARM,
        IncidentStatus.NOT_CONFIRMED,
        IncidentStatus.CLOSED,
    }


def test_enum_values_match_srs_codes() -> None:
    assert IncidentStatus("NEW") is IncidentStatus.NEW
    assert IncidentType("zone_intrusion") is IncidentType.ZONE_INTRUSION
    # SRS §12.2 stores severity and role in uppercase.
    assert Severity("CRITICAL") is Severity.CRITICAL
    assert Role("HEAD_OF_SECURITY") is Role.HEAD_OF_SECURITY


def test_every_incident_type_has_a_severity() -> None:
    for incident_type in IncidentType:
        assert severity_for(incident_type) in Severity


def test_severity_matches_srs() -> None:
    assert severity_for(IncidentType.ZONE_INTRUSION) is Severity.MEDIUM
    assert severity_for(IncidentType.FALL) is Severity.HIGH
    assert severity_for(IncidentType.FIGHT) is Severity.CRITICAL
    assert severity_for(IncidentType.FIRE) is Severity.CRITICAL


def test_severity_rank_orders_critical_first() -> None:
    shuffled = [Severity.MEDIUM, Severity.CRITICAL, Severity.HIGH]
    result = sorted(shuffled, key=lambda s: SEVERITY_RANK[s])
    assert result == [Severity.CRITICAL, Severity.HIGH, Severity.MEDIUM]


def test_every_severity_has_a_rank() -> None:
    for severity in Severity:
        assert severity in SEVERITY_RANK
