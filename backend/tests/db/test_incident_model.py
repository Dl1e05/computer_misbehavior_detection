import uuid
from datetime import UTC, datetime

import pytest
from sqlalchemy import select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Camera, Incident, Participant
from app.domain.camera import CameraSourceType
from app.domain.incident import ClipState, IncidentStatus, IncidentType, Severity


async def add_camera(db_session: AsyncSession) -> uuid.UUID:
    camera = Camera(
        code="cam-01",
        name="Corridor",
        source_type=CameraSourceType.RTSP,
        source_ref="rtsp://media-server/cam-01",
    )
    db_session.add(camera)
    await db_session.flush()
    return camera.id


def make_incident(camera_id: uuid.UUID, **changes: object) -> Incident:
    incident = Incident(
        incident_type=IncidentType.FALL,
        severity=Severity.HIGH,
        status=IncidentStatus.NEW,
        camera_id=camera_id,
        occurred_at=datetime(2026, 1, 1, tzinfo=UTC),
    )
    for name, value in changes.items():
        setattr(incident, name, value)
    return incident


async def test_incident_gets_database_defaults(db_session: AsyncSession) -> None:
    incident = make_incident(await add_camera(db_session))
    db_session.add(incident)
    await db_session.flush()
    await db_session.refresh(incident)

    assert incident.clip_state is ClipState.PENDING
    assert incident.created_received_at.tzinfo is not None
    assert incident.version == 1
    assert incident.zone_id is None
    assert incident.measures is None


async def test_severity_must_match_incident_type(db_session: AsyncSession) -> None:
    # A fall is HIGH (SRS §5); the database refuses any other severity.
    db_session.add(
        make_incident(await add_camera(db_session), severity=Severity.MEDIUM)
    )
    with pytest.raises(IntegrityError, match="ck_incidents_severity_matches_type"):
        await db_session.flush()


@pytest.mark.parametrize(
    "measures",
    [pytest.param(None, id="no measures"), pytest.param("   ", id="blank measures")],
)
async def test_closed_incident_needs_measures(
    db_session: AsyncSession, measures: str | None
) -> None:
    # I7 (SRS §12.5).
    db_session.add(
        make_incident(
            await add_camera(db_session),
            status=IncidentStatus.CLOSED,
            measures=measures,
        )
    )
    with pytest.raises(IntegrityError, match="ck_incidents_closed_requires_measures"):
        await db_session.flush()


async def test_closed_incident_with_measures_is_accepted(
    db_session: AsyncSession,
) -> None:
    incident = make_incident(
        await add_camera(db_session),
        status=IncidentStatus.CLOSED,
        measures="Guard sent to the corridor",
    )
    db_session.add(incident)
    await db_session.flush()

    assert incident.id is not None


async def test_occurred_at_is_required(db_session: AsyncSession) -> None:
    # No default on purpose: the time must come from the ML event.
    db_session.add(make_incident(await add_camera(db_session), occurred_at=None))
    with pytest.raises(IntegrityError, match="occurred_at"):
        await db_session.flush()


async def test_incident_needs_an_existing_camera(db_session: AsyncSession) -> None:
    db_session.add(make_incident(camera_id=uuid.uuid4()))
    with pytest.raises(IntegrityError, match="fk_incidents_camera_id_cameras"):
        await db_session.flush()


async def add_incident(db_session: AsyncSession, camera_id: uuid.UUID) -> uuid.UUID:
    incident = make_incident(camera_id)
    db_session.add(incident)
    await db_session.flush()
    return incident.id


def make_participant(incident_id: uuid.UUID, **changes: object) -> Participant:
    participant = Participant(
        incident_id=incident_id,
        track_id="track-1",
        confidence=0.9,
        bbox=[10, 20, 100, 200],
        model_version="test-model-1",
    )
    for name, value in changes.items():
        setattr(participant, name, value)
    return participant


async def test_participant_round_trips(db_session: AsyncSession) -> None:
    incident_id = await add_incident(db_session, await add_camera(db_session))
    participant = make_participant(incident_id)
    db_session.add(participant)
    await db_session.flush()

    participant_id = participant.id
    db_session.expire_all()
    stored = await db_session.scalar(
        select(Participant).where(Participant.id == participant_id)
    )
    assert stored is not None
    assert stored.bbox == [10, 20, 100, 200]
    assert stored.confidence == 0.9
    assert stored.snapshot_ref is None


async def test_track_is_unique_within_an_incident(db_session: AsyncSession) -> None:
    incident_id = await add_incident(db_session, await add_camera(db_session))
    db_session.add(make_participant(incident_id))
    await db_session.flush()

    db_session.add(make_participant(incident_id))
    with pytest.raises(IntegrityError, match="uq_participants_incident_id_track_id"):
        await db_session.flush()


async def test_same_track_in_another_incident_is_accepted(
    db_session: AsyncSession,
) -> None:
    camera_id = await add_camera(db_session)
    first = await add_incident(db_session, camera_id)
    second = await add_incident(db_session, camera_id)

    db_session.add_all([make_participant(first), make_participant(second)])
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
            {"bbox": [10, 20, 100]}, "ck_participants_bbox_has_4_values", id="3 values"
        ),
    ],
)
async def test_invalid_participant_is_rejected(
    db_session: AsyncSession, changes: dict[str, object], constraint: str
) -> None:
    incident_id = await add_incident(db_session, await add_camera(db_session))
    db_session.add(make_participant(incident_id, **changes))
    with pytest.raises(IntegrityError, match=constraint):
        await db_session.flush()


async def test_participants_are_removed_with_their_incident(
    db_session: AsyncSession,
) -> None:
    incident_id = await add_incident(db_session, await add_camera(db_session))
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
