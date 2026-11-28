import asyncio
import uuid
from collections.abc import AsyncIterator, Sequence
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import select, text
from sqlalchemy.exc import OperationalError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.config import Settings
from app.db.models import AuditEntry, Camera, Escalation, Incident, Timer
from app.db.session import build_engine, create_session_factory
from app.domain.camera import CameraSourceType
from app.domain.incident import (
    IncidentStatus,
    IncidentType,
    Severity,
    TimerKind,
    TimerState,
)
from app.services import timers
from app.services.timers import fire_due_timers

T0 = datetime(2026, 10, 25, 9, 0, tzinfo=UTC)
DUE = T0 + timedelta(seconds=10)  # every test timer is due at this moment


@pytest.fixture
async def sessions(db_session: AsyncSession) -> async_sessionmaker[AsyncSession]:
    # The service commits; joining the test's transaction rolls it back after.
    return async_sessionmaker(
        bind=await db_session.connection(),
        join_transaction_mode="create_savepoint",
        expire_on_commit=False,
    )


async def stored[M](session: AsyncSession, model: type[M]) -> Sequence[M]:
    """Rows as the database has them, not as the session remembers them."""
    statement = select(model).execution_options(populate_existing=True)
    return (await session.scalars(statement)).all()


async def make(
    session: AsyncSession,
    camera_id: uuid.UUID,
    status: IncidentStatus,
    kind: TimerKind,
    *,
    deadline_at: datetime = DUE,
) -> tuple[Incident, Timer]:
    """An incident with one pending timer."""
    # Suspicion timers belong to fights; the severity must match the type
    # (database CHECK), so FIGHT is CRITICAL and FALL is HIGH.
    if kind is TimerKind.SUSPICION_TIMEOUT:
        incident_type, severity = IncidentType.FIGHT, Severity.CRITICAL
    else:
        incident_type, severity = IncidentType.FALL, Severity.HIGH
    incident = Incident(
        incident_type=incident_type,
        severity=severity,
        status=status,
        camera_id=camera_id,
        occurred_at=T0,
        created_received_at=T0,
        model_version="test-model-1",
    )
    session.add(incident)
    await session.flush()  # the database creates incident.id

    timer = Timer(incident_id=incident.id, kind=kind, deadline_at=deadline_at)
    session.add(timer)
    await session.flush()
    return incident, timer


# T4: SUSPECTED -> NOT_CONFIRMED after 10 s (BE-FR-24)


async def test_unconfirmed_suspicion_times_out(
    db_session: AsyncSession,
    camera_id: uuid.UUID,
    sessions: async_sessionmaker[AsyncSession],
) -> None:
    await make(
        db_session, camera_id, IncidentStatus.SUSPECTED, TimerKind.SUSPICION_TIMEOUT
    )

    assert await fire_due_timers(sessions, DUE) == 1

    (incident,) = await stored(db_session, Incident)
    assert incident.status is IncidentStatus.NOT_CONFIRMED
    (timer,) = await stored(db_session, Timer)
    assert timer.state is TimerState.FIRED
    assert timer.fired_at == DUE
    (entry,) = await stored(db_session, AuditEntry)
    assert entry.action == "incident.transition"
    assert entry.actor_id is None
    assert entry.access_context == {
        "timer_id": str(timer.id),
        "trigger": "TIMER",
        "from_status": "SUSPECTED",
        "to_status": "NOT_CONFIRMED",
    }


async def test_timer_before_its_deadline_does_not_fire(
    db_session: AsyncSession,
    camera_id: uuid.UUID,
    sessions: async_sessionmaker[AsyncSession],
) -> None:
    await make(
        db_session, camera_id, IncidentStatus.SUSPECTED, TimerKind.SUSPICION_TIMEOUT
    )

    assert await fire_due_timers(sessions, DUE - timedelta(microseconds=1)) == 0

    (incident,) = await stored(db_session, Incident)
    assert incident.status is IncidentStatus.SUSPECTED
    (timer,) = await stored(db_session, Timer)
    assert timer.state is TimerState.PENDING


async def test_suspicion_timer_of_a_confirmed_fight_is_cancelled(
    db_session: AsyncSession,
    camera_id: uuid.UUID,
    sessions: async_sessionmaker[AsyncSession],
) -> None:
    # Normally T3 cancels it; the timer must still never undo a confirmation.
    await make(db_session, camera_id, IncidentStatus.NEW, TimerKind.SUSPICION_TIMEOUT)

    assert await fire_due_timers(sessions, DUE) == 0

    (incident,) = await stored(db_session, Incident)
    assert incident.status is IncidentStatus.NEW
    (timer,) = await stored(db_session, Timer)
    assert timer.state is TimerState.CANCELLED
    assert timer.fired_at is None
    assert await stored(db_session, AuditEntry) == []


# Escalation after 30 s in NEW (BE-FR-26, BE-FR-27)


async def test_incident_left_in_new_escalates(
    db_session: AsyncSession,
    camera_id: uuid.UUID,
    sessions: async_sessionmaker[AsyncSession],
) -> None:
    await make(db_session, camera_id, IncidentStatus.NEW, TimerKind.ESCALATION)

    assert await fire_due_timers(sessions, DUE) == 1

    (escalation,) = await stored(db_session, Escalation)
    assert escalation.level == 1
    assert escalation.raised_at == DUE
    assert escalation.acknowledged_at is None
    # A notification, not a status change (SRS §5.3).
    (incident,) = await stored(db_session, Incident)
    assert incident.status is IncidentStatus.NEW
    (timer,) = await stored(db_session, Timer)
    assert timer.state is TimerState.FIRED
    (entry,) = await stored(db_session, AuditEntry)
    assert entry.action == "incident.escalated"
    assert entry.access_context == {
        "timer_id": str(timer.id),
        "trigger": "TIMER",
        "level": 1,
        "severity": "HIGH",
    }


@pytest.mark.parametrize(
    "status",
    [
        IncidentStatus.IN_PROGRESS,
        IncidentStatus.CONFIRMED,
        IncidentStatus.FALSE_ALARM,
    ],
)
async def test_incident_taken_into_work_does_not_escalate(
    db_session: AsyncSession,
    camera_id: uuid.UUID,
    sessions: async_sessionmaker[AsyncSession],
    status: IncidentStatus,
) -> None:
    # T-TMR-02: the operator was faster than the timer.
    await make(db_session, camera_id, status, TimerKind.ESCALATION)

    assert await fire_due_timers(sessions, DUE) == 0

    (timer,) = await stored(db_session, Timer)
    assert timer.state is TimerState.CANCELLED
    assert await stored(db_session, Escalation) == []
    assert await stored(db_session, AuditEntry) == []


# Firing once, late, in order


async def test_timer_fires_only_once(
    db_session: AsyncSession,
    camera_id: uuid.UUID,
    sessions: async_sessionmaker[AsyncSession],
) -> None:
    await make(db_session, camera_id, IncidentStatus.NEW, TimerKind.ESCALATION)

    assert await fire_due_timers(sessions, DUE) == 1
    assert await fire_due_timers(sessions, DUE + timedelta(seconds=1)) == 0
    assert len(await stored(db_session, Escalation)) == 1


async def test_late_timer_fires_at_once(
    db_session: AsyncSession,
    camera_id: uuid.UUID,
    sessions: async_sessionmaker[AsyncSession],
) -> None:
    # T-TMR-03: the worker was down past the deadline; nothing is lost.
    await make(db_session, camera_id, IncidentStatus.NEW, TimerKind.ESCALATION)
    restarted_at = DUE + timedelta(hours=1)

    assert await fire_due_timers(sessions, restarted_at) == 1

    (timer,) = await stored(db_session, Timer)
    assert timer.fired_at == restarted_at


async def test_batch_takes_the_earliest_deadlines_first(
    db_session: AsyncSession,
    camera_id: uuid.UUID,
    sessions: async_sessionmaker[AsyncSession],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(timers, "BATCH", 2)
    deadlines = [DUE, DUE - timedelta(seconds=2), DUE - timedelta(seconds=1)]
    for deadline in deadlines:
        await make(
            db_session,
            camera_id,
            IncidentStatus.NEW,
            TimerKind.ESCALATION,
            deadline_at=deadline,
        )

    assert await fire_due_timers(sessions, DUE) == 2
    pending = [
        timer for timer in await stored(db_session, Timer) if timer.fired_at is None
    ]
    assert [timer.deadline_at for timer in pending] == [DUE]
    # The rest waits for the next tick.
    assert await fire_due_timers(sessions, DUE) == 1


async def test_timer_cancelled_after_the_list_was_read_does_not_fire(
    db_session: AsyncSession,
    camera_id: uuid.UUID,
    sessions: async_sessionmaker[AsyncSession],
) -> None:
    # The race the re-check after locking is for: the list is read without
    # locks, then a confirmation (T3) cancels the timer before it is fired.
    incident, timer = await make(
        db_session, camera_id, IncidentStatus.NEW, TimerKind.ESCALATION
    )
    timer.state = TimerState.CANCELLED
    await db_session.flush()

    assert not await timers._fire_one(sessions, timer.id, incident.id, DUE)
    assert await stored(db_session, Escalation) == []


# Failures


async def test_one_broken_timer_does_not_stop_the_others(
    db_session: AsyncSession,
    camera_id: uuid.UUID,
    sessions: async_sessionmaker[AsyncSession],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, broken = await make(
        db_session,
        camera_id,
        IncidentStatus.NEW,
        TimerKind.ESCALATION,
        deadline_at=DUE - timedelta(seconds=1),  # first in line
    )
    _, healthy = await make(
        db_session, camera_id, IncidentStatus.NEW, TimerKind.ESCALATION
    )
    real_fire_one = timers._fire_one

    async def fail_for_one(
        session_factory: async_sessionmaker[AsyncSession],
        timer_id: uuid.UUID,
        incident_id: uuid.UUID,
        now: datetime,
    ) -> bool:
        if timer_id == broken.id:
            raise RuntimeError("bug")
        return await real_fire_one(session_factory, timer_id, incident_id, now)

    monkeypatch.setattr(timers, "_fire_one", fail_for_one)

    assert await fire_due_timers(sessions, DUE) == 1
    states = {timer.id: timer.state for timer in await stored(db_session, Timer)}
    assert states == {broken.id: TimerState.PENDING, healthy.id: TimerState.FIRED}


async def test_database_outage_reaches_the_worker(
    db_session: AsyncSession,
    camera_id: uuid.UUID,
    sessions: async_sessionmaker[AsyncSession],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Not swallowed per timer: the worker must see it and back off.
    await make(db_session, camera_id, IncidentStatus.NEW, TimerKind.ESCALATION)

    async def database_down(*args: object) -> bool:
        raise OperationalError("UPDATE ...", {}, Exception("connection lost"))

    monkeypatch.setattr(timers, "_fire_one", database_down)

    with pytest.raises(OperationalError):
        await fire_due_timers(sessions, DUE)


# Locks (two real connections, so the rows are committed for real)


@pytest.fixture
async def committed(migrated_db: Settings) -> AsyncIterator[Settings]:
    """Settings for tests that commit; the tables are emptied after."""
    try:
        yield migrated_db
    finally:
        # As the owner: the app may not delete audit entries (BE-FR-75).
        owner = create_async_engine(migrated_db.migration_database_url)
        try:
            async with owner.begin() as connection:
                await connection.execute(
                    text("TRUNCATE cameras, audit_entries CASCADE")
                )
        finally:
            await owner.dispose()


async def test_incident_locked_by_ingest_is_skipped_until_free(
    committed: Settings,
) -> None:
    # Ingest holds the incident (a confirmation is being saved). The timer
    # must not wait for it, or the two could deadlock: it tries next tick.
    engine = build_engine(committed)
    factory = create_session_factory(engine)
    try:
        async with factory() as session:
            camera = Camera(
                code="cam-lock",
                name="Camera",
                source_type=CameraSourceType.RTSP,
                source_ref="rtsp://media-server/cam-lock",
            )
            session.add(camera)
            await session.flush()
            incident, _ = await make(
                session, camera.id, IncidentStatus.NEW, TimerKind.ESCALATION
            )
            await session.commit()

        async with factory() as ingest:
            await ingest.execute(
                select(Incident).where(Incident.id == incident.id).with_for_update()
            )
            # Returns at once instead of waiting for the lock; the timeout
            # turns a wait into a failure instead of a hanging test run.
            async with asyncio.timeout(5):
                assert await fire_due_timers(factory, DUE) == 0
            await ingest.commit()  # lock released

        assert await fire_due_timers(factory, DUE) == 1
    finally:
        await engine.dispose()
