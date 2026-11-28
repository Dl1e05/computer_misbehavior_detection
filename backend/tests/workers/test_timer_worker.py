"""The timer worker loop against a real database (SRS §6.5)."""

import asyncio
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import func, select, text
from sqlalchemy.exc import OperationalError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.config import Settings
from app.db.models import Camera, Escalation, Incident, Timer
from app.db.session import build_engine, create_session_factory
from app.domain.camera import CameraSourceType
from app.domain.incident import IncidentStatus, IncidentType, Severity, TimerKind
from app.services.timers import fire_due_timers
from app.workers import timers as worker

TIMEOUT = 10.0


@pytest.fixture
async def sessions(
    migrated_db: Settings,
) -> AsyncIterator[async_sessionmaker[AsyncSession]]:
    # The worker commits for real, so the tables are emptied after.
    engine = build_engine(migrated_db)
    try:
        yield create_session_factory(engine)
    finally:
        await engine.dispose()
        owner = create_async_engine(migrated_db.migration_database_url)
        try:
            async with owner.begin() as connection:
                await connection.execute(
                    text("TRUNCATE cameras, audit_entries CASCADE")
                )
        finally:
            await owner.dispose()


async def add_due_escalation(sessions: async_sessionmaker[AsyncSession]) -> None:
    now = datetime.now(UTC)
    async with sessions() as session:
        camera = Camera(
            code="cam-01",
            name="Camera 1",
            source_type=CameraSourceType.RTSP,
            source_ref="rtsp://media-server/cam-01",
        )
        session.add(camera)
        await session.flush()
        incident = Incident(
            incident_type=IncidentType.FALL,
            severity=Severity.HIGH,
            status=IncidentStatus.NEW,
            camera_id=camera.id,
            occurred_at=now,
            created_received_at=now,
            model_version="test-model-1",
        )
        session.add(incident)
        await session.flush()
        session.add(
            Timer(
                incident_id=incident.id,
                kind=TimerKind.ESCALATION,
                deadline_at=now - timedelta(seconds=1),
            )
        )
        await session.commit()


async def escalations(sessions: async_sessionmaker[AsyncSession]) -> int:
    async with sessions() as session:
        return await session.scalar(select(func.count()).select_from(Escalation)) or 0


async def run_until_escalated(
    settings: Settings, sessions: async_sessionmaker[AsyncSession]
) -> None:
    task = asyncio.create_task(worker.run(settings))
    try:
        async with asyncio.timeout(TIMEOUT):
            while await escalations(sessions) == 0:  # noqa: ASYNC110
                await asyncio.sleep(0.05)
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


async def test_worker_fires_due_timers(
    migrated_db: Settings,
    sessions: async_sessionmaker[AsyncSession],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(worker, "TICK", 0.05)
    await add_due_escalation(sessions)

    await run_until_escalated(migrated_db, sessions)

    assert await escalations(sessions) == 1


async def test_worker_keeps_going_while_the_database_is_down(
    migrated_db: Settings,
    sessions: async_sessionmaker[AsyncSession],
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setattr(worker, "TICK", 0.01)
    failures = 2

    async def down_twice(*args: object, **kwargs: object) -> int:
        nonlocal failures
        if failures:
            failures -= 1
            raise OperationalError("SELECT ...", {}, Exception("connection lost"))
        return await fire_due_timers(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr("app.workers.timers.fire_due_timers", down_twice)
    await add_due_escalation(sessions)

    await run_until_escalated(migrated_db, sessions)

    # Logged and survived: the timer still fired once the database was back.
    warnings = [r for r in caplog.records if "not reachable" in r.getMessage()]
    assert len(warnings) == 2


async def test_worker_stops_on_a_bug(
    migrated_db: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def bug(*args: object, **kwargs: object) -> int:
        raise RuntimeError("bug outside any single timer")

    monkeypatch.setattr("app.workers.timers.fire_due_timers", bug)

    with pytest.raises(RuntimeError, match="bug outside"):
        async with asyncio.timeout(TIMEOUT):
            await worker.run(migrated_db)


async def test_worker_tries_a_broken_timer_only_once(
    migrated_db: Settings,
    sessions: async_sessionmaker[AsyncSession],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Otherwise it would be retried, and logged, on every tick forever.
    monkeypatch.setattr(worker, "TICK", 0.01)
    tries = 0

    async def always_fails(*args: object) -> bool:
        nonlocal tries
        tries += 1
        raise RuntimeError("bug")

    monkeypatch.setattr("app.services.timers._fire_one", always_fails)
    await add_due_escalation(sessions)

    task = asyncio.create_task(worker.run(migrated_db))
    try:
        await asyncio.sleep(0.5)  # dozens of ticks
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    assert tries == 1
