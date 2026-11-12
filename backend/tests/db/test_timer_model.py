import uuid
from datetime import UTC, datetime

import pytest
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Camera, Incident, Timer
from app.domain.camera import CameraSourceType
from app.domain.incident import (
    IncidentStatus,
    IncidentType,
    Severity,
    TimerKind,
    TimerState,
)

DEADLINE = datetime(2026, 1, 1, 0, 0, 30, tzinfo=UTC)
FIRED_AT = datetime(2026, 1, 1, 0, 0, 31, tzinfo=UTC)


async def add_incident(db_session: AsyncSession) -> uuid.UUID:
    camera = Camera(
        code="cam-01",
        name="Corridor",
        source_type=CameraSourceType.RTSP,
        source_ref="rtsp://media-server/cam-01",
    )
    db_session.add(camera)
    await db_session.flush()
    incident = Incident(
        incident_type=IncidentType.FIGHT,
        severity=Severity.CRITICAL,
        status=IncidentStatus.SUSPECTED,
        camera_id=camera.id,
        occurred_at=datetime(2026, 1, 1, tzinfo=UTC),
    )
    db_session.add(incident)
    await db_session.flush()
    return incident.id


def make_timer(incident_id: uuid.UUID, **changes: object) -> Timer:
    timer = Timer(
        incident_id=incident_id,
        kind=TimerKind.ESCALATION,
        deadline_at=DEADLINE,
    )
    for name, value in changes.items():
        setattr(timer, name, value)
    return timer


async def test_timer_gets_database_defaults(db_session: AsyncSession) -> None:
    timer = make_timer(await add_incident(db_session))
    db_session.add(timer)
    await db_session.flush()
    await db_session.refresh(timer)

    assert timer.state is TimerState.PENDING
    assert timer.fired_at is None
    assert timer.version == 1


async def test_second_pending_timer_of_same_kind_is_rejected(
    db_session: AsyncSession,
) -> None:
    incident_id = await add_incident(db_session)
    db_session.add(make_timer(incident_id))
    await db_session.flush()

    db_session.add(make_timer(incident_id))
    with pytest.raises(IntegrityError, match="uq_timers_one_pending_per_kind"):
        await db_session.flush()


async def test_new_pending_timer_is_accepted_after_old_one_fired(
    db_session: AsyncSession,
) -> None:
    # Proves the unique index is partial: finished timers stay as history
    # and do not block a new one.
    incident_id = await add_incident(db_session)
    first = make_timer(incident_id)
    db_session.add(first)
    await db_session.flush()

    first.state = TimerState.FIRED
    first.fired_at = FIRED_AT
    await db_session.flush()

    db_session.add(make_timer(incident_id))
    await db_session.flush()


async def test_new_pending_timer_is_accepted_after_old_one_cancelled(
    db_session: AsyncSession,
) -> None:
    incident_id = await add_incident(db_session)
    first = make_timer(incident_id)
    db_session.add(first)
    await db_session.flush()

    first.state = TimerState.CANCELLED
    await db_session.flush()

    db_session.add(make_timer(incident_id))
    await db_session.flush()


async def test_pending_timers_of_different_kinds_are_accepted(
    db_session: AsyncSession,
) -> None:
    incident_id = await add_incident(db_session)
    db_session.add_all(
        [
            make_timer(incident_id, kind=TimerKind.SUSPICION_TIMEOUT),
            make_timer(incident_id, kind=TimerKind.ESCALATION),
        ]
    )
    await db_session.flush()


@pytest.mark.parametrize(
    "changes",
    [
        pytest.param({"state": TimerState.FIRED}, id="fired without fired_at"),
        pytest.param({"fired_at": FIRED_AT}, id="pending with fired_at"),
        pytest.param(
            {"state": TimerState.CANCELLED, "fired_at": FIRED_AT},
            id="cancelled with fired_at",
        ),
    ],
)
async def test_fired_at_is_set_only_for_fired_timers(
    db_session: AsyncSession, changes: dict[str, object]
) -> None:
    db_session.add(make_timer(await add_incident(db_session), **changes))
    with pytest.raises(IntegrityError, match="ck_timers_fired_at_iff_fired"):
        await db_session.flush()
