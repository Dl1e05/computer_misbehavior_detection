import hashlib
import uuid
from datetime import UTC, datetime

import pytest
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import DeadLetter, ProcessedEvent

PAYLOAD_HASH = hashlib.sha256(b"{}").hexdigest()


def make_processed_event(**changes: object) -> ProcessedEvent:
    event = ProcessedEvent(
        event_id=uuid.UUID("5f0c2a7e-3b9d-4c1a-9e8f-2d7b6a1c0e42"),
        payload_hash=PAYLOAD_HASH,
        result={"outcome": "created", "incident_id": "some-id"},
    )
    for name, value in changes.items():
        setattr(event, name, value)
    return event


def make_dead_letter(**changes: object) -> DeadLetter:
    dead_letter = DeadLetter(
        received_at=datetime(2026, 1, 1, tzinfo=UTC),
        topic="videoanalytics/v1/cameras/cam-02/events",
        failure_code="INVALID_EVENT_TYPE",
        payload_hash=PAYLOAD_HASH,
        raw_payload={"camera_id": "cam-02", "event_type": "telepathy"},
    )
    for name, value in changes.items():
        setattr(dead_letter, name, value)
    return dead_letter


async def test_processed_event_round_trips(db_session: AsyncSession) -> None:
    event = make_processed_event()
    db_session.add(event)
    await db_session.flush()

    event_id = event.event_id
    db_session.expire_all()
    stored = await db_session.scalar(
        select(ProcessedEvent).where(ProcessedEvent.event_id == event_id)
    )
    assert stored is not None
    assert stored.result == {"outcome": "created", "incident_id": "some-id"}
    assert stored.processed_at.tzinfo is not None


async def test_same_event_id_is_stored_once(db_session: AsyncSession) -> None:
    # Redelivery safety (V16): the primary key itself refuses a second row.
    db_session.add(make_processed_event())
    await db_session.flush()
    db_session.expunge_all()

    db_session.add(make_processed_event())
    with pytest.raises(IntegrityError, match="pk_processed_events"):
        await db_session.flush()


BAD_HASHES = [
    pytest.param("abc", id="too short"),
    pytest.param("0" * 65, id="too long"),
    pytest.param(PAYLOAD_HASH.upper(), id="uppercase"),
    pytest.param("z" * 64, id="not hex"),
    pytest.param("sha256:" + PAYLOAD_HASH[:57], id="prefixed"),
]


@pytest.mark.parametrize("payload_hash", BAD_HASHES)
async def test_processed_event_hash_must_be_lowercase_sha256_hex(
    db_session: AsyncSession, payload_hash: str
) -> None:
    db_session.add(make_processed_event(payload_hash=payload_hash))
    with pytest.raises(IntegrityError, match="ck_processed_events_payload_hash"):
        await db_session.flush()


@pytest.mark.parametrize("payload_hash", BAD_HASHES)
async def test_dead_letter_hash_must_be_lowercase_sha256_hex(
    db_session: AsyncSession, payload_hash: str
) -> None:
    # Same format in both tables, so the two can be matched by hash.
    db_session.add(make_dead_letter(payload_hash=payload_hash))
    with pytest.raises(IntegrityError, match="ck_dead_letters_payload_hash"):
        await db_session.flush()


@pytest.mark.parametrize(
    "result",
    [
        pytest.param(None, id="None becomes JSON null"),
        pytest.param([], id="list"),
        pytest.param("created", id="string"),
    ],
)
async def test_processed_event_result_must_be_an_object(
    db_session: AsyncSession, result: object
) -> None:
    db_session.add(make_processed_event(result=result))
    with pytest.raises(IntegrityError, match="ck_processed_events_result_is_object"):
        await db_session.flush()


async def test_dead_letter_payload_round_trips(db_session: AsyncSession) -> None:
    dead_letter = make_dead_letter()
    db_session.add(dead_letter)
    await db_session.flush()

    dead_letter_id = dead_letter.id
    db_session.expire_all()
    stored = await db_session.scalar(
        select(DeadLetter).where(DeadLetter.id == dead_letter_id)
    )
    assert stored is not None
    assert stored.raw_payload == {"camera_id": "cam-02", "event_type": "telepathy"}


async def test_dead_letter_without_payload_is_accepted(
    db_session: AsyncSession,
) -> None:
    # A MALFORMED_JSON payload may contain embeddings that cannot be
    # removed, so ingest keeps only its hash (SRS §12.6, §14).
    db_session.add(make_dead_letter(failure_code="MALFORMED_JSON", raw_payload=None))
    await db_session.flush()


async def test_dead_letter_received_at_is_required(db_session: AsyncSession) -> None:
    # No default on purpose: ingest sets it when the message arrives.
    db_session.add(make_dead_letter(received_at=None))
    with pytest.raises(IntegrityError, match="received_at"):
        await db_session.flush()
