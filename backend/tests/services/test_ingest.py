import hashlib
import json
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import AuditEntry, Camera, DeadLetter, ProcessedEvent, Zone
from app.domain.audit import AuditObjectType, AuditResult
from app.domain.camera import CameraSourceType
from app.domain.event import FailureCode
from app.mqtt.events import EventLimits
from app.services.ingest import IngestOutcome, IngestStatus, ingest_event

DIM = 4
RECEIVED_AT = datetime(2026, 10, 25, 9, 14, 4, tzinfo=UTC)
LIMITS = EventLimits(
    embedding_dim=DIM, received_at=RECEIVED_AT, max_future=timedelta(seconds=60)
)
TOPIC = "videoanalytics/v1/cameras/cam-01/events"
EVENT_ID = "5f0c2a7e-3b9d-4c1a-9e8f-2d7b6a1c0e42"


def event(**changes: Any) -> dict[str, Any]:
    """A valid detection for the camera_id fixture's camera (cam-01)."""
    return {
        "schema_version": "1.0",
        "event_id": EVENT_ID,
        "event_type": "detection",
        "camera_id": "cam-01",
        "occurred_at": "2026-10-25T09:14:03.217Z",
        "incident_type": "fall",
        "model_version": "fall-1.0",
        "participants": [{"track_id": "17", "confidence": 0.91}],
    } | changes


def fight_with_embedding(event_type: str, length: int = DIM) -> dict[str, Any]:
    payload = event(event_type=event_type, incident_type="fight")
    payload["participants"][0] |= {
        "embedding": [0.123456789] * length,
        "embedding_model_version": "insightface-buffalo_l-1.0",
    }
    return payload


def encode(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload).encode()


async def ingest(session: AsyncSession, raw: bytes) -> IngestOutcome:
    return await ingest_event(session, raw, topic=TOPIC, limits=LIMITS)


async def count(session: AsyncSession, model: type[Any]) -> int:
    return await session.scalar(select(func.count()).select_from(model)) or 0


async def only_dead_letter(session: AsyncSession) -> DeadLetter:
    return (await session.scalars(select(DeadLetter))).one()


async def add_camera(session: AsyncSession, code: str) -> Camera:
    camera = Camera(
        code=code,
        name=code,
        source_type=CameraSourceType.RTSP,
        source_ref=f"rtsp://media-server/{code}",
    )
    session.add(camera)
    await session.flush()
    return camera


async def add_zone(session: AsyncSession, camera_id: uuid.UUID) -> Zone:
    zone = Zone(
        camera_id=camera_id,
        name="Door",
        polygon=[[0, 0], [10, 0], [10, 10]],
        frame_width=1920,
        frame_height=1080,
    )
    session.add(zone)
    await session.flush()
    return zone


# Accepted


@pytest.mark.usefixtures("camera_id")
async def test_valid_event_is_accepted(db_session: AsyncSession) -> None:
    raw = encode(event())
    outcome = await ingest(db_session, raw)

    assert outcome == IngestOutcome(
        IngestStatus.ACCEPTED, result={"outcome": "accepted"}
    )
    stored = await db_session.get(ProcessedEvent, uuid.UUID(EVENT_ID))
    assert stored is not None
    assert stored.payload_hash == hashlib.sha256(raw).hexdigest()
    assert await count(db_session, DeadLetter) == 0


async def test_zone_intrusion_with_own_active_zone_is_accepted(
    db_session: AsyncSession, zone_id: uuid.UUID
) -> None:
    raw = encode(event(incident_type="zone_intrusion", zone_id=str(zone_id)))
    assert (await ingest(db_session, raw)).status is IngestStatus.ACCEPTED


# V16


@pytest.mark.usefixtures("camera_id")
async def test_redelivery_returns_stored_result(db_session: AsyncSession) -> None:
    raw = encode(event())
    first = await ingest(db_session, raw)
    second = await ingest(db_session, raw)

    assert second == IngestOutcome(IngestStatus.DUPLICATE, result=first.result)
    assert await count(db_session, ProcessedEvent) == 1
    assert await count(db_session, DeadLetter) == 0


@pytest.mark.usefixtures("camera_id")
async def test_same_event_id_with_other_payload_is_a_conflict(
    db_session: AsyncSession,
) -> None:
    first_raw = encode(event())
    await ingest(db_session, first_raw)

    outcome = await ingest(db_session, encode(event(model_version="fall-2.0")))

    assert outcome.code is FailureCode.EVENT_ID_CONFLICT
    stored = await db_session.get(ProcessedEvent, uuid.UUID(EVENT_ID))
    assert stored is not None
    assert stored.payload_hash == hashlib.sha256(first_raw).hexdigest()
    assert (await only_dead_letter(db_session)).failure_code == "EVENT_ID_CONFLICT"


async def test_redelivery_after_camera_was_disabled_is_still_a_duplicate(
    db_session: AsyncSession, camera_id: uuid.UUID
) -> None:
    raw = encode(event())
    await ingest(db_session, raw)
    camera = await db_session.get(Camera, camera_id)
    assert camera is not None
    camera.enabled = False
    await db_session.flush()

    assert (await ingest(db_session, raw)).status is IngestStatus.DUPLICATE


# Rejections and dead letters


@pytest.mark.usefixtures("camera_id")
async def test_malformed_message_is_kept_as_hash_only(
    db_session: AsyncSession,
) -> None:
    raw = b"not json"
    outcome = await ingest(db_session, raw)

    assert outcome == IngestOutcome(
        IngestStatus.REJECTED, code=FailureCode.MALFORMED_JSON
    )
    dead_letter = await only_dead_letter(db_session)
    assert dead_letter.failure_code == "MALFORMED_JSON"
    assert dead_letter.payload_hash == hashlib.sha256(raw).hexdigest()
    assert dead_letter.raw_payload is None
    assert dead_letter.topic == TOPIC
    assert dead_letter.received_at == RECEIVED_AT
    assert await count(db_session, ProcessedEvent) == 0


@pytest.mark.usefixtures("camera_id")
async def test_srs_invalid_fixture_reports_first_failure(
    db_session: AsyncSession,
) -> None:
    # SRS §7.7 (T-EV-01): several faults, V4 comes first.
    payload = {
        "schema_version": "1.0",
        "event_id": "not-a-uuid",
        "event_type": "suspicion",
        "camera_id": "cam-09",
        "occurred_at": "25.10.2026 09:14",
        "incident_type": "fall",
        "model_version": "fall-pose-1.0.0",
        "participants": [
            {
                "track_id": "17",
                "confidence": 0.77,
                "embedding": [0.1, 0.2],
                "embedding_model_version": "insightface-buffalo_l-1.0",
            }
        ],
    }
    outcome = await ingest(db_session, encode(payload))

    assert outcome.code is FailureCode.INVALID_EVENT_ID
    dead_letter = await only_dead_letter(db_session)
    assert dead_letter.raw_payload is not None
    assert "embedding" not in dead_letter.raw_payload["participants"][0]


async def test_unregistered_camera_is_rejected(db_session: AsyncSession) -> None:
    outcome = await ingest(db_session, encode(event(camera_id="cam-09")))

    assert outcome.code is FailureCode.UNKNOWN_CAMERA
    dead_letter = await only_dead_letter(db_session)
    assert dead_letter.failure_code == "UNKNOWN_CAMERA"
    # The diagnostic needs the camera id (§7.5).
    assert dead_letter.raw_payload is not None
    assert dead_letter.raw_payload["camera_id"] == "cam-09"


async def test_disabled_camera_is_rejected(
    db_session: AsyncSession, camera_id: uuid.UUID
) -> None:
    camera = await db_session.get(Camera, camera_id)
    assert camera is not None
    camera.enabled = False
    await db_session.flush()

    outcome = await ingest(db_session, encode(event()))
    assert outcome.code is FailureCode.UNKNOWN_CAMERA


@pytest.mark.usefixtures("camera_id")
async def test_known_deviation_camera_checked_after_parse(
    db_session: AsyncSession,
) -> None:
    # V8 is checked after parse_event, so a far-future time (V9) next to an
    # unknown camera is reported as V9 (see ingest_event). Pinned on purpose.
    payload = event(camera_id="cam-09", occurred_at="2026-10-25T10:00:00Z")
    outcome = await ingest(db_session, encode(payload))
    assert outcome.code is FailureCode.INVALID_TIMESTAMP


@pytest.mark.usefixtures("camera_id")
async def test_nul_in_camera_id_is_rejected_not_crashing(
    db_session: AsyncSession,
) -> None:
    # Postgres would refuse it in the camera lookup; the message would then
    # never be acknowledged.
    outcome = await ingest(db_session, encode(event(camera_id="cam-01\x00")))

    assert outcome.code is FailureCode.UNKNOWN_CAMERA
    assert (await only_dead_letter(db_session)).raw_payload is None


@pytest.mark.parametrize(
    "zone",
    [
        pytest.param("zone-1", id="not a uuid"),
        pytest.param("", id="empty"),
        pytest.param(str(uuid.uuid4()), id="does not exist"),
        pytest.param("inactive", id="inactive"),
        pytest.param("other camera", id="of another camera"),
    ],
)
async def test_unusable_zone_is_rejected(
    db_session: AsyncSession, camera_id: uuid.UUID, zone_id: uuid.UUID, zone: str
) -> None:
    if zone == "inactive":
        own = await db_session.get(Zone, zone_id)
        assert own is not None
        own.active = False
        await db_session.flush()
        zone = str(zone_id)
    elif zone == "other camera":
        other = await add_camera(db_session, "cam-02")
        zone = str((await add_zone(db_session, other.id)).id)

    payload = event(incident_type="zone_intrusion", zone_id=zone)
    outcome = await ingest(db_session, encode(payload))
    assert outcome.code is FailureCode.INVALID_ZONE


async def test_zone_is_checked_on_any_incident_type(
    db_session: AsyncSession, camera_id: uuid.UUID
) -> None:
    other = await add_camera(db_session, "cam-02")
    foreign_zone = await add_zone(db_session, other.id)

    outcome = await ingest(db_session, encode(event(zone_id=str(foreign_zone.id))))
    assert outcome.code is FailureCode.INVALID_ZONE


# V14 and embeddings


@pytest.mark.usefixtures("camera_id")
async def test_embedding_on_suspicion_is_a_privacy_violation(
    db_session: AsyncSession,
) -> None:
    outcome = await ingest(db_session, encode(fight_with_embedding("suspicion")))

    assert outcome.code is FailureCode.BIOMETRIC_NOT_ALLOWED
    dead_letter = await only_dead_letter(db_session)
    assert dead_letter.raw_payload is not None
    assert "0.123456789" not in json.dumps(dead_letter.raw_payload)
    participant = dead_letter.raw_payload["participants"][0]
    assert participant["embedding_model_version"] == "insightface-buffalo_l-1.0"

    audit = (await db_session.scalars(select(AuditEntry))).one()
    assert audit.action == "event.privacy_violation"
    assert audit.result is AuditResult.DENIED
    assert audit.actor_id is None
    assert audit.object_type is AuditObjectType.EVENT
    assert audit.object_id == EVENT_ID
    assert audit.access_context == {"topic": TOPIC, "camera_id": "cam-01"}


@pytest.mark.usefixtures("camera_id")
async def test_embedding_is_never_kept_in_any_dead_letter(
    db_session: AsyncSession,
) -> None:
    # Wrong length (V15) on an event where embeddings are allowed.
    payload = fight_with_embedding("confirmation", length=DIM - 1)
    outcome = await ingest(db_session, encode(payload))

    assert outcome.code is FailureCode.INVALID_EMBEDDING
    dead_letter = await only_dead_letter(db_session)
    assert "0.123456789" not in json.dumps(dead_letter.raw_payload)
    # Only the privacy violation is audited, not every rejection.
    assert await count(db_session, AuditEntry) == 0


@pytest.mark.usefixtures("camera_id")
async def test_dead_letter_with_nan_is_written(db_session: AsyncSession) -> None:
    payload = event()
    payload["participants"][0]["confidence"] = float("nan")
    outcome = await ingest(db_session, encode(payload))

    assert outcome.status is IngestStatus.REJECTED
    assert (await only_dead_letter(db_session)).raw_payload is None
