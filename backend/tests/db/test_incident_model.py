import uuid
from datetime import UTC, datetime

import pytest
from sqlalchemy import select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Incident, Participant
from app.domain.incident import (
    SEVERITY_BY_TYPE,
    ClipState,
    IncidentStatus,
    IncidentType,
    Severity,
)


def make_incident(camera_id: uuid.UUID, **changes: object) -> Incident:
    incident = Incident(
        incident_type=IncidentType.FALL,
        severity=Severity.HIGH,
        status=IncidentStatus.NEW,
        camera_id=camera_id,
        occurred_at=datetime(2026, 1, 1, tzinfo=UTC),
        created_received_at=datetime(2026, 1, 1, 0, 0, 1, tzinfo=UTC),
    )
    for name, value in changes.items():
        setattr(incident, name, value)
    return incident


async def test_incident_gets_database_defaults(
    db_session: AsyncSession, camera_id: uuid.UUID
) -> None:
    incident = make_incident(camera_id)
    db_session.add(incident)
    await db_session.flush()
    await db_session.refresh(incident)

    assert incident.clip_state is ClipState.PENDING
    assert incident.version == 1
    assert incident.zone_id is None
    assert incident.measures is None


async def test_severity_must_match_incident_type(
    db_session: AsyncSession, camera_id: uuid.UUID
) -> None:
    # A fall is HIGH (SRS §5); the database refuses any other severity.
    db_session.add(make_incident(camera_id, severity=Severity.MEDIUM))
    with pytest.raises(IntegrityError, match="ck_incidents_severity_matches_type"):
        await db_session.flush()


async def test_severity_check_matches_domain_map(
    db_session: AsyncSession, camera_id: uuid.UUID, zone_id: uuid.UUID
) -> None:
    # Autogenerate does not compare CHECK bodies, so a change to
    # SEVERITY_BY_TYPE without a migration would only show up here.
    accepted = set()
    for incident_type in IncidentType:
        for severity in Severity:
            try:
                async with db_session.begin_nested():
                    db_session.add(
                        make_incident(
                            camera_id,
                            incident_type=incident_type,
                            severity=severity,
                            zone_id=zone_id,
                        )
                    )
            except IntegrityError:
                continue
            accepted.add((incident_type, severity))

    assert accepted == set(SEVERITY_BY_TYPE.items())


async def test_zone_intrusion_needs_a_zone(
    db_session: AsyncSession, camera_id: uuid.UUID
) -> None:
    # V11 (SRS §7.4).
    db_session.add(
        make_incident(
            camera_id,
            incident_type=IncidentType.ZONE_INTRUSION,
            severity=Severity.MEDIUM,
        )
    )
    with pytest.raises(IntegrityError, match="ck_incidents_zone_intrusion_requires"):
        await db_session.flush()


async def test_zone_intrusion_with_a_zone_is_accepted(
    db_session: AsyncSession, camera_id: uuid.UUID, zone_id: uuid.UUID
) -> None:
    db_session.add(
        make_incident(
            camera_id,
            incident_type=IncidentType.ZONE_INTRUSION,
            severity=Severity.MEDIUM,
            zone_id=zone_id,
        )
    )
    await db_session.flush()


@pytest.mark.parametrize(
    "measures",
    [
        pytest.param(None, id="no measures"),
        pytest.param("   ", id="spaces"),
        pytest.param("\n", id="newline"),
        pytest.param(" \t\r\n ", id="mixed whitespace"),
    ],
)
async def test_closed_incident_needs_measures(
    db_session: AsyncSession, camera_id: uuid.UUID, measures: str | None
) -> None:
    # I7 (SRS §12.5).
    db_session.add(
        make_incident(
            camera_id,
            status=IncidentStatus.CLOSED,
            measures=measures,
        )
    )
    with pytest.raises(IntegrityError, match="ck_incidents_closed_requires_measures"):
        await db_session.flush()


async def test_closed_incident_with_measures_is_accepted(
    db_session: AsyncSession,
    camera_id: uuid.UUID,
) -> None:
    incident = make_incident(
        camera_id,
        status=IncidentStatus.CLOSED,
        measures="Guard sent to the corridor",
    )
    db_session.add(incident)
    await db_session.flush()

    assert incident.id is not None


async def test_occurred_at_is_required(
    db_session: AsyncSession, camera_id: uuid.UUID
) -> None:
    # No default on purpose: the time must come from the ML event.
    db_session.add(make_incident(camera_id, occurred_at=None))
    with pytest.raises(IntegrityError, match="occurred_at"):
        await db_session.flush()


async def test_received_at_is_required(
    db_session: AsyncSession, camera_id: uuid.UUID
) -> None:
    # No default on purpose: ingest sets it when the message arrives.
    db_session.add(make_incident(camera_id, created_received_at=None))
    with pytest.raises(IntegrityError, match="created_received_at"):
        await db_session.flush()


async def test_incident_needs_an_existing_camera(db_session: AsyncSession) -> None:
    db_session.add(make_incident(camera_id=uuid.uuid4()))
    with pytest.raises(IntegrityError, match="fk_incidents_camera_id_cameras"):
        await db_session.flush()


def make_participant(incident_id: uuid.UUID, **changes: object) -> Participant:
    participant = Participant(
        incident_id=incident_id,
        track_id="track-1",
        confidence=0.9,
        bbox=[412.0, 188.5, 498.0, 402.0],
        model_version="test-model-1",
    )
    for name, value in changes.items():
        setattr(participant, name, value)
    return participant


async def test_participant_round_trips(
    db_session: AsyncSession, incident_id: uuid.UUID
) -> None:
    participant = make_participant(incident_id)
    db_session.add(participant)
    await db_session.flush()

    participant_id = participant.id
    db_session.expire_all()
    stored = await db_session.scalar(
        select(Participant).where(Participant.id == participant_id)
    )
    assert stored is not None
    # The SRS sample bbox has fractional pixels.
    assert stored.bbox == [412.0, 188.5, 498.0, 402.0]
    assert stored.confidence == 0.9
    assert stored.snapshot_ref is None


async def test_participant_without_bbox_is_accepted(
    db_session: AsyncSession, incident_id: uuid.UUID
) -> None:
    # bbox is optional in the event contract (SRS §7.3).
    db_session.add(make_participant(incident_id, bbox=None))
    await db_session.flush()


async def test_track_is_unique_within_an_incident(
    db_session: AsyncSession, incident_id: uuid.UUID
) -> None:
    db_session.add(make_participant(incident_id))
    await db_session.flush()

    db_session.add(make_participant(incident_id))
    with pytest.raises(IntegrityError, match="uq_participants_incident_id_track_id"):
        await db_session.flush()


async def test_same_track_in_another_incident_is_accepted(
    db_session: AsyncSession,
    camera_id: uuid.UUID,
    incident_id: uuid.UUID,
) -> None:
    second = make_incident(camera_id)
    db_session.add(second)
    await db_session.flush()

    db_session.add_all([make_participant(incident_id), make_participant(second.id)])
    await db_session.flush()


@pytest.mark.parametrize(
    ("changes", "constraint"),
    [
        pytest.param(
            {"confidence": 1.5}, "ck_participants_confidence_in_range", id="conf > 1"
        ),
        pytest.param(
            {"confidence": -0.1}, "ck_participants_confidence_in_range", id="conf < 0"
        ),
        pytest.param(
            {"bbox": [10.0, 20.0, 100.0]},
            "ck_participants_bbox_has_4_values",
            id="3 values",
        ),
    ],
)
async def test_invalid_participant_is_rejected(
    db_session: AsyncSession,
    incident_id: uuid.UUID,
    changes: dict[str, object],
    constraint: str,
) -> None:
    db_session.add(make_participant(incident_id, **changes))
    with pytest.raises(IntegrityError, match=constraint):
        await db_session.flush()


async def test_participants_are_removed_with_their_incident(
    db_session: AsyncSession,
    incident_id: uuid.UUID,
) -> None:
    db_session.add(make_participant(incident_id))
    await db_session.flush()

    await db_session.execute(
        text("DELETE FROM incidents WHERE id = :id"), {"id": incident_id}
    )

    remaining = await db_session.scalar(
        text("SELECT count(*) FROM participants WHERE incident_id = :id"),
        {"id": incident_id},
    )
    assert remaining == 0
