from app.domain.incident import (
    TERMINAL_STATUSES,
    IncidentStatus,
    IncidentType,
    Role,
    Severity,
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
