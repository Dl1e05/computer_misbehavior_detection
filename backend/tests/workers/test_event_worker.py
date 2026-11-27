"""The event worker against a real broker and database (SRS §7.1, T-RES-05).

The worker commits for real, so these tests do not use the rolled-back
db_session; every test starts and ends with empty tables.
"""

import asyncio
import itertools
import json
import time
import uuid
from collections.abc import AsyncIterator, Callable, Coroutine, Iterator
from datetime import datetime
from typing import Any

import aiomqtt
import pytest
from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from testcontainers.community.mqtt import MosquittoContainer

from app.config import Settings, get_settings
from app.db.models import Camera, Incident, ProcessedEvent
from app.db.session import (
    UnsafeDatabaseRoleError,
    build_engine,
    check_app_role_is_limited,
    create_session_factory,
)
from app.domain.camera import CameraSourceType
from app.mqtt.consumer import Ack, EventMessageHandler
from app.services.ingest import IngestOutcome, ingest_event
from app.workers import events as worker

TOPIC = "videoanalytics/v1/cameras/cam-01/events"
# Long enough for a slow CI machine; a passing test returns much earlier.
TIMEOUT = 15.0


@pytest.fixture(scope="module")
def broker() -> Iterator[MosquittoContainer]:
    # The same image as compose.dev.yaml.
    with MosquittoContainer("eclipse-mosquitto:2") as container:
        yield container


@pytest.fixture
async def settings(
    migrated_db: Settings,
    broker: MosquittoContainer,
    monkeypatch: pytest.MonkeyPatch,
) -> AsyncIterator[Settings]:
    monkeypatch.setenv("MQTT_HOST", broker.get_container_host_ip())
    monkeypatch.setenv("MQTT_PORT", str(broker.get_exposed_port(1883)))
    # Own session per test: no messages left over from another test.
    monkeypatch.setenv("MQTT_CLIENT_ID", f"be-events-{uuid.uuid4().hex[:8]}")
    monkeypatch.setattr(worker, "FIRST_DELAY", 0.1)
    get_settings.cache_clear()
    settings = get_settings()

    await _empty_tables(settings)
    try:
        yield settings
    finally:
        await _empty_tables(settings)


async def _empty_tables(settings: Settings) -> None:
    # As the owner: the app may not delete audit entries (BE-FR-75).
    owner = create_async_engine(settings.migration_database_url)
    try:
        async with owner.begin() as connection:
            await connection.execute(
                text(
                    "TRUNCATE cameras, audit_entries, dead_letters, "
                    "processed_events CASCADE"
                )
            )
    finally:
        await owner.dispose()


@pytest.fixture
async def sessions(
    settings: Settings,
) -> AsyncIterator[async_sessionmaker[AsyncSession]]:
    engine = build_engine(settings)
    factory = create_session_factory(engine)
    async with factory() as session:
        session.add(
            Camera(
                code="cam-01",
                name="Camera 1",
                source_type=CameraSourceType.RTSP,
                source_ref="rtsp://media-server/cam-01",
            )
        )
        await session.commit()
    try:
        yield factory
    finally:
        await engine.dispose()


@pytest.fixture
def outcomes(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """The status of every message the worker processes, in order."""
    seen: list[str] = []

    async def recording(*args: Any, **kwargs: Any) -> IngestOutcome:
        outcome = await ingest_event(*args, **kwargs)
        seen.append(outcome.status.value)
        return outcome

    monkeypatch.setattr("app.mqtt.consumer.ingest_event", recording)
    return seen


def fall() -> bytes:
    return json.dumps(
        {
            "schema_version": "1.0",
            "event_id": str(uuid.uuid4()),
            "event_type": "detection",
            "camera_id": "cam-01",
            # The worker sets received_at to now; any past time is valid.
            "occurred_at": "2026-10-25T09:14:03.217Z",
            "incident_type": "fall",
            "run_id": "run-1",
            "model_version": "fall-1.0",
            "participants": [{"track_id": 1, "confidence": 0.9}],
        }
    ).encode()


async def publish(settings: Settings, *payloads: bytes) -> None:
    """Publishes like ML: QoS 1, returns once the broker has each message."""
    async with aiomqtt.Client(
        settings.mqtt_host,
        settings.mqtt_port,
        identifier=f"ml-{uuid.uuid4().hex[:8]}",
    ) as client:
        for payload in payloads:
            await client.publish(TOPIC, payload, qos=1)


class Worker:
    """Starts and stops worker.run like a process would be."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        self._task = asyncio.create_task(worker.run(self._settings))
        # Subscribed once the broker knows the session; publishing earlier
        # would reach no one on the very first start.
        await asyncio.sleep(0.5)

    async def stop(self) -> None:
        assert self._task is not None
        self._task.cancel()
        await asyncio.gather(self._task, return_exceptions=True)

    @property
    def task(self) -> asyncio.Task[None]:
        assert self._task is not None
        return self._task


@pytest.fixture
async def running(settings: Settings) -> AsyncIterator[Worker]:
    started = Worker(settings)
    try:
        yield started
    finally:
        if started._task is not None:
            await started.stop()


async def count(sessions: async_sessionmaker[AsyncSession], model: type[Any]) -> int:
    async with sessions() as session:
        return await session.scalar(select(func.count()).select_from(model)) or 0


async def wait_until(condition: Callable[[], Coroutine[Any, Any, bool]]) -> None:
    # Polling is right here: the condition is a database row, which no
    # asyncio.Event can signal.
    async with asyncio.timeout(TIMEOUT):
        while not await condition():  # noqa: ASYNC110
            await asyncio.sleep(0.1)


async def processed(sessions: async_sessionmaker[AsyncSession], expected: int) -> None:
    async def done() -> bool:
        return await count(sessions, ProcessedEvent) == expected

    await wait_until(done)


async def test_event_is_saved_and_acknowledged(
    settings: Settings,
    sessions: async_sessionmaker[AsyncSession],
    running: Worker,
    outcomes: list[str],
) -> None:
    await running.start()
    await publish(settings, fall())
    await processed(sessions, 1)
    assert await count(sessions, Incident) == 1

    # Acknowledged: a restart does not get it again.
    await running.stop()
    await running.start()
    await asyncio.sleep(1)
    assert outcomes == ["accepted"]


async def test_events_published_while_offline_are_delivered(
    settings: Settings,
    sessions: async_sessionmaker[AsyncSession],
    running: Worker,
    outcomes: list[str],
) -> None:
    # T-RES-05: the broker keeps them for the persistent session.
    await running.start()
    await running.stop()

    await publish(settings, fall(), fall())
    await running.start()

    await processed(sessions, 2)
    assert outcomes == ["accepted", "accepted"]


async def test_unsaved_message_comes_again(
    settings: Settings,
    sessions: async_sessionmaker[AsyncSession],
    running: Worker,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The first try is "not saved"; the worker reconnects and the broker
    # sends it again, followed by the next message.
    real = EventMessageHandler.handle
    tries: list[bytes] = []

    async def first_try_fails(
        self: EventMessageHandler, topic: str, payload: bytes, received_at: datetime
    ) -> Ack:
        tries.append(payload)
        if len(tries) == 1:
            return Ack.RETRY
        return await real(self, topic, payload, received_at)

    monkeypatch.setattr(EventMessageHandler, "handle", first_try_fails)
    first, second = fall(), fall()
    await running.start()
    await publish(settings, first, second)

    await processed(sessions, 2)
    assert tries[:2] == [first, first]
    assert await count(sessions, Incident) == 2


async def test_crash_before_acknowledgement_gives_a_duplicate(
    settings: Settings,
    sessions: async_sessionmaker[AsyncSession],
    running: Worker,
    outcomes: list[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Saved and committed, but the process dies before the PUBACK.
    def crash(client: aiomqtt.Client, message: aiomqtt.Message) -> None:
        raise RuntimeError("crash before ack")

    real_acknowledge = worker.acknowledge
    monkeypatch.setattr(worker, "acknowledge", crash)
    await running.start()
    await publish(settings, fall())
    with pytest.raises(RuntimeError, match="crash before ack"):
        await asyncio.wait_for(running.task, TIMEOUT)

    monkeypatch.setattr(worker, "acknowledge", real_acknowledge)
    await running.start()

    async def got_again() -> bool:
        return outcomes == ["accepted", "duplicate"]

    await wait_until(got_again)
    assert await count(sessions, Incident) == 1


async def test_retry_delay_grows_and_resets_after_progress(
    settings: Settings,
    sessions: async_sessionmaker[AsyncSession],
    running: Worker,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Connecting is not progress: while a message keeps failing, the pause
    # doubles (0.1, 0.2, 0.4 s). A saved message starts it again at 0.1 s.
    monkeypatch.setattr(worker, "MAX_DELAY", 0.4)
    real = EventMessageHandler.handle
    first, second = fall(), fall()
    tries: dict[bytes, list[float]] = {first: [], second: []}
    failures = {first: 4, second: 1}

    async def fails_at_first(
        self: EventMessageHandler, topic: str, payload: bytes, received_at: datetime
    ) -> Ack:
        tries[payload].append(time.monotonic())
        if len(tries[payload]) <= failures[payload]:
            return Ack.RETRY
        return await real(self, topic, payload, received_at)

    monkeypatch.setattr(EventMessageHandler, "handle", fails_at_first)
    await running.start()
    await publish(settings, first, second)
    await processed(sessions, 2)

    pauses = [later - earlier for earlier, later in itertools.pairwise(tries[first])]
    # Each pause also holds a reconnect, so compare them with each other.
    assert pauses[2] > pauses[0] + 0.2  # 0.4 s against 0.1 s
    after_progress = tries[second][1] - tries[second][0]
    assert after_progress < pauses[2] - 0.15


# Without a broker


async def test_worker_waits_for_the_database(
    migrated_db: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Started before Postgres is ready: retried, not a crash.
    monkeypatch.setattr(worker, "FIRST_DELAY", 0.01)
    tries = 0

    async def not_ready_twice(engine: AsyncEngine) -> None:
        nonlocal tries
        tries += 1
        if tries <= 2:
            raise ConnectionRefusedError
        await check_app_role_is_limited(engine)

    monkeypatch.setattr("app.workers.events.check_app_role_is_limited", not_ready_twice)
    engine = build_engine(migrated_db)
    try:
        await worker.wait_for_database(engine)
    finally:
        await engine.dispose()
    assert tries == 3


async def test_worker_stops_on_an_unsafe_database_login(
    migrated_db: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def unsafe(engine: AsyncEngine) -> None:
        raise UnsafeDatabaseRoleError("superuser")

    monkeypatch.setattr("app.workers.events.check_app_role_is_limited", unsafe)
    monkeypatch.setattr(worker, "FIRST_DELAY", 0.01)
    engine = build_engine(migrated_db)
    try:
        # A timeout, so waiting forever fails instead of hanging the run.
        with pytest.raises(UnsafeDatabaseRoleError):
            async with asyncio.timeout(TIMEOUT):
                await worker.wait_for_database(engine)
    finally:
        await engine.dispose()
