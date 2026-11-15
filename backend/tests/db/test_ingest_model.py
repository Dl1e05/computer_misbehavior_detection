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
        failure_code="MALFORMED_JSON",
        payload_hash=PAYLOAD_HASH,
        raw_payload=b"{}",
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


@pytest.mark.parametrize(
    "payload_hash",
    [pytest.param("abc", id="too short"), pytest.param("0" * 65, id="too long")],
)
async def test_payload_hash_must_be_sha256_hex(
    db_session: AsyncSession, payload_hash: str
) -> None:
    db_session.add(make_processed_event(payload_hash=payload_hash))
    with pytest.raises(IntegrityError, match="ck_processed_events_payload_hash"):
        await db_session.flush()


async def test_dead_letter_keeps_payload_that_is_not_json(
    db_session: AsyncSession,
) -> None:
    # Why bytea and not JSONB: a MALFORMED_JSON payload (V1) may not even be
    # valid UTF-8.
    garbage = b"\xff\xfe{not json"
    dead_letter = make_dead_letter(raw_payload=garbage)
    db_session.add(dead_letter)
    await db_session.flush()

    dead_letter_id = dead_letter.id
    db_session.expire_all()
    stored = await db_session.scalar(
        select(DeadLetter).where(DeadLetter.id == dead_letter_id)
    )
    assert stored is not None
    assert stored.raw_payload == garbage


async def test_dead_letter_received_at_is_required(db_session: AsyncSession) -> None:
    # No default on purpose: ingest sets it when the message arrives.
    db_session.add(make_dead_letter(received_at=None))
    with pytest.raises(IntegrityError, match="received_at"):
        await db_session.flush()
