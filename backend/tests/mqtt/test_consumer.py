import hashlib
import json
import logging
import uuid
from collections.abc import Callable, Coroutine
from datetime import UTC, datetime
from typing import Any

import pytest
from sqlalchemy import func, select
from sqlalchemy.exc import DBAPIError, IntegrityError, OperationalError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.config import Settings
from app.db.models import DeadLetter, Incident, ProcessedEvent
from app.mqtt.consumer import MAX_ATTEMPTS, Ack, EventMessageHandler
from app.services.ingest import IngestOutcome, ingest_event

RECEIVED_AT = datetime(2026, 10, 25, 9, 14, 4, tzinfo=UTC)
TOPIC = "videoanalytics/v1/cameras/cam-01/events"
SECRET = "run-0.123456789"


def payload(**changes: Any) -> bytes:
    """A valid fall for the camera_id fixture's camera (cam-01)."""
    return json.dumps(
        {
            "schema_version": "1.0",
            "event_id": str(uuid.uuid4()),
            "event_type": "detection",
            "camera_id": "cam-01",
            "occurred_at": "2026-10-25T09:14:03.217Z",
            "incident_type": "fall",
            "run_id": SECRET,
            "model_version": "fall-1.0",
            "participants": [{"track_id": 17, "confidence": 0.91}],
        }
        | changes
    ).encode()


@pytest.fixture
async def handler(
    db_session: AsyncSession, migrated_db: Settings
) -> EventMessageHandler:
    # Sessions join the test's transaction, so their commits only release a
    # savepoint and the test still rolls everything back.
    factory = async_sessionmaker(
        bind=await db_session.connection(),
        join_transaction_mode="create_savepoint",
        expire_on_commit=False,
    )
    return EventMessageHandler(factory, migrated_db)


Ingest = Callable[..., Coroutine[Any, Any, IngestOutcome]]


def failing(*errors: Exception) -> Callable[[Ingest], Ingest]:
    """Wraps ingest_event: raises the given errors in turn, then works."""
    remaining = list(errors)

    def wrap(real: Ingest) -> Ingest:
        async def fake(*args: Any, **kwargs: Any) -> IngestOutcome:
            if remaining:
                raise remaining.pop(0)
            return await real(*args, **kwargs)

        return fake

    return wrap


def patch_ingest(
    monkeypatch: pytest.MonkeyPatch, wrap: Callable[[Ingest], Ingest]
) -> None:
    monkeypatch.setattr("app.mqtt.consumer.ingest_event", wrap(ingest_event))


async def count(session: AsyncSession, model: type[Any]) -> int:
    return await session.scalar(select(func.count()).select_from(model)) or 0


def database_error(cls: type[DBAPIError], *, invalidated: bool = False) -> DBAPIError:
    # Shaped like a real error: its message holds the SQL parameters.
    return cls(
        "INSERT ...",
        {"run_id": SECRET},
        Exception("boom"),
        connection_invalidated=invalidated,
    )


async def handle(handler: EventMessageHandler, raw: bytes) -> Ack:
    return await handler.handle(TOPIC, raw, RECEIVED_AT)


# Saved messages are acknowledged


@pytest.mark.usefixtures("camera_id")
async def test_accepted_event_is_committed_and_acknowledged(
    db_session: AsyncSession, handler: EventMessageHandler
) -> None:
    assert await handle(handler, payload()) is Ack.ACK

    # Visible from another session: the handler committed.
    assert await count(db_session, ProcessedEvent) == 1
    assert await count(db_session, Incident) == 1


@pytest.mark.usefixtures("camera_id")
async def test_rejected_event_is_acknowledged(
    db_session: AsyncSession, handler: EventMessageHandler
) -> None:
    # A rejection is saved too (a dead letter), so the message is done.
    assert await handle(handler, b"not json") is Ack.ACK
    assert await count(db_session, DeadLetter) == 1


@pytest.mark.usefixtures("camera_id")
async def test_received_at_is_the_given_time(
    db_session: AsyncSession, handler: EventMessageHandler
) -> None:
    await handle(handler, payload())
    incident = (await db_session.scalars(select(Incident))).one()
    assert incident.created_received_at == RECEIVED_AT


# Lost connection: retried without limit


@pytest.mark.usefixtures("camera_id")
@pytest.mark.parametrize(
    "error",
    [
        pytest.param(ConnectionRefusedError(), id="connection refused"),
        pytest.param(database_error(OperationalError), id="operational error"),
        pytest.param(
            database_error(DBAPIError, invalidated=True), id="connection lost"
        ),
    ],
)
async def test_database_unavailable_is_retried_without_limit(
    db_session: AsyncSession,
    handler: EventMessageHandler,
    monkeypatch: pytest.MonkeyPatch,
    error: Exception,
) -> None:
    attempts = MAX_ATTEMPTS + 2
    patch_ingest(monkeypatch, failing(*[error] * attempts))
    raw = payload()

    for _ in range(attempts):
        assert await handle(handler, raw) is Ack.RETRY

    # Never given up: once the database is back, the alarm is saved.
    assert await count(db_session, DeadLetter) == 0
    assert await handle(handler, raw) is Ack.ACK
    assert await count(db_session, Incident) == 1


# Other errors: given up after MAX_ATTEMPTS


@pytest.mark.usefixtures("camera_id")
@pytest.mark.parametrize(
    "error",
    [
        pytest.param(RuntimeError("bug"), id="bug"),
        pytest.param(database_error(DBAPIError), id="value Postgres refuses"),
    ],
)
async def test_failing_message_is_given_up_as_a_dead_letter(
    db_session: AsyncSession,
    handler: EventMessageHandler,
    monkeypatch: pytest.MonkeyPatch,
    error: Exception,
) -> None:
    patch_ingest(monkeypatch, failing(*[error] * MAX_ATTEMPTS))
    raw = payload()

    for _ in range(MAX_ATTEMPTS - 1):
        assert await handle(handler, raw) is Ack.RETRY
        assert await count(db_session, DeadLetter) == 0
    assert await handle(handler, raw) is Ack.ACK

    dead_letter = (await db_session.scalars(select(DeadLetter))).one()
    assert dead_letter.failure_code == "PROCESSING_ERROR"
    assert dead_letter.payload_hash == hashlib.sha256(raw).hexdigest()
    assert dead_letter.raw_payload is None
    assert dead_letter.topic == TOPIC
    assert dead_letter.received_at == RECEIVED_AT
    assert await count(db_session, ProcessedEvent) == 0


@pytest.mark.usefixtures("camera_id")
async def test_race_passes_on_the_next_try(
    db_session: AsyncSession,
    handler: EventMessageHandler,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Two first events of one fight: the loser hits the unique index.
    patch_ingest(monkeypatch, failing(database_error(IntegrityError)))
    raw = payload()

    assert await handle(handler, raw) is Ack.RETRY
    assert await handle(handler, raw) is Ack.ACK
    assert await count(db_session, Incident) == 1


@pytest.mark.usefixtures("camera_id")
async def test_success_resets_the_count(
    db_session: AsyncSession,
    handler: EventMessageHandler,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = payload(event_id=str(uuid.uuid4()))
    bug = RuntimeError("bug")
    patch_ingest(monkeypatch, failing(*[bug] * (MAX_ATTEMPTS - 1)))
    for _ in range(MAX_ATTEMPTS - 1):
        await handle(handler, raw)
    assert await handle(handler, raw) is Ack.ACK  # saved

    # The same bytes fail again later (a redelivery, already processed
    # here, would succeed; the count must start again either way).
    patch_ingest(monkeypatch, failing(bug))
    assert await handle(handler, raw) is Ack.RETRY
    assert await count(db_session, DeadLetter) == 0


@pytest.mark.usefixtures("camera_id")
async def test_attempts_are_counted_per_message(
    db_session: AsyncSession,
    handler: EventMessageHandler,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bug = RuntimeError("bug")
    patch_ingest(monkeypatch, failing(*[bug] * (MAX_ATTEMPTS * 2)))
    first, second = payload(), payload()

    for _ in range(MAX_ATTEMPTS - 1):
        assert await handle(handler, first) is Ack.RETRY
        assert await handle(handler, second) is Ack.RETRY
    assert await count(db_session, DeadLetter) == 0


@pytest.mark.usefixtures("camera_id")
async def test_dead_letter_that_fails_keeps_the_message(
    db_session: AsyncSession,
    handler: EventMessageHandler,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    patch_ingest(monkeypatch, failing(*[RuntimeError("bug")] * (MAX_ATTEMPTS + 1)))
    real_add = AsyncSession.add

    def refuse_dead_letters(self: AsyncSession, instance: object, **kw: Any) -> None:
        if isinstance(instance, DeadLetter):
            raise ConnectionRefusedError
        real_add(self, instance, **kw)

    monkeypatch.setattr(AsyncSession, "add", refuse_dead_letters)
    raw = payload()
    for _ in range(MAX_ATTEMPTS - 1):
        await handle(handler, raw)

    # Not acknowledged, so not lost; the next try gives up again at once.
    assert await handle(handler, raw) is Ack.RETRY
    monkeypatch.setattr(AsyncSession, "add", real_add)
    assert await handle(handler, raw) is Ack.ACK
    assert await count(db_session, DeadLetter) == 1


# Logs


@pytest.mark.usefixtures("camera_id")
@pytest.mark.parametrize(
    "error",
    [
        pytest.param(database_error(DBAPIError), id="database error"),
        pytest.param(database_error(OperationalError), id="database unavailable"),
    ],
)
async def test_logs_never_contain_event_values(
    handler: EventMessageHandler,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    error: Exception,
) -> None:
    patch_ingest(monkeypatch, failing(*[error] * MAX_ATTEMPTS))
    raw = payload()
    with caplog.at_level(logging.INFO, logger="app.mqtt.consumer"):
        for _ in range(MAX_ATTEMPTS):
            await handle(handler, raw)

    assert caplog.records
    assert SECRET not in caplog.text
