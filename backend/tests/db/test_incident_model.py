import uuid
from datetime import UTC, datetime

import pytest
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Camera, Incident
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
