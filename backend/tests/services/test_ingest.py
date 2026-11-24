import hashlib
import json
import math
import uuid
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import AuditEntry, Camera, DeadLetter, ProcessedEvent, Zone
from app.domain.audit import AuditObjectType, AuditResult
from app.domain.event import FailureCode
from app.mqtt.events import EventLimits
from app.services.ingest import IngestOutcome, IngestStatus, ingest_event

RECEIVED_AT = datetime(2026, 10, 25, 9, 14, 4, tzinfo=UTC)
LIMITS = EventLimits(received_at=RECEIVED_AT, max_future=timedelta(seconds=60))
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
        "run_id": "run-1",
        "model_version": "fall-1.0",
        "participants": [{"track_id": 17, "confidence": 0.91}],
    } | changes


def event_with_embedding(**changes: Any) -> dict[str, Any]:
    payload = event(**changes)
    payload["participants"][0] |= {
        "embedding": [0.123456789] * 4,
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


# The make_camera / make_zone fixtures (tests/conftest.py); test modules cannot
# import conftest types, so they are spelled out here.
MakeCamera = Callable[[str], Awaitable[Camera]]
MakeZone = Callable[[uuid.UUID], Awaitable[Zone]]


async def only_audit_entry(session: AsyncSession) -> AuditEntry:
    return (await session.scalars(select(AuditEntry))).one()


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
    # SRS §7.7 (T-EV-01): several faults, V4 comes first. run_id and the
    # integer track_id were agreed with ML later; without them V2 is first.
    payload = {
        "schema_version": "1.0",
        "event_id": "not-a-uuid",
        "event_type": "suspicion",
        "camera_id": "cam-09",
        "occurred_at": "25.10.2026 09:14",
        "incident_type": "fall",
        "run_id": "run-1",
        "model_version": "fall-pose-1.0.0",
        "participants": [
            {
                "track_id": 17,
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
    db_session: AsyncSession,
    camera_id: uuid.UUID,
    zone_id: uuid.UUID,
    make_camera: MakeCamera,
    make_zone: MakeZone,
    zone: str,
) -> None:
    if zone == "inactive":
        own = await db_session.get(Zone, zone_id)
        assert own is not None
        own.active = False
        await db_session.flush()
        zone = str(zone_id)
    elif zone == "other camera":
        other = await make_camera("cam-02")
        zone = str((await make_zone(other.id)).id)

    payload = event(incident_type="zone_intrusion", zone_id=zone)
    outcome = await ingest(db_session, encode(payload))
    assert outcome.code is FailureCode.INVALID_ZONE


@pytest.mark.usefixtures("camera_id")
async def test_zone_is_checked_on_any_incident_type(
    db_session: AsyncSession, make_camera: MakeCamera, make_zone: MakeZone
) -> None:
    other = await make_camera("cam-02")
    foreign_zone = await make_zone(other.id)

    outcome = await ingest(db_session, encode(event(zone_id=str(foreign_zone.id))))
    assert outcome.code is FailureCode.INVALID_ZONE


# V14 and face data


@pytest.mark.usefixtures("camera_id")
@pytest.mark.parametrize(
    ("event_type", "incident_type"),
    [("suspicion", "fight"), ("confirmation", "fight"), ("detection", "fall")],
)
async def test_embedding_in_any_event_is_a_privacy_violation(
    db_session: AsyncSession, event_type: str, incident_type: str
) -> None:
    payload = event_with_embedding(event_type=event_type, incident_type=incident_type)
    outcome = await ingest(db_session, encode(payload))

    assert outcome.code is FailureCode.BIOMETRIC_NOT_ALLOWED
    dead_letter = await only_dead_letter(db_session)
    assert dead_letter.raw_payload is not None
    assert "0.123456789" not in json.dumps(dead_letter.raw_payload)
    # The rest of the message stays, for diagnostics.
    assert dead_letter.raw_payload["participants"] == [
        {"track_id": 17, "confidence": 0.91}
    ]

    audit = await only_audit_entry(db_session)
    assert audit.action == "event.privacy_violation"
    assert audit.result is AuditResult.DENIED
    assert audit.actor_id is None
    assert audit.object_type is AuditObjectType.EVENT
    assert audit.object_id == EVENT_ID
    assert audit.access_context == {"topic": TOPIC, "camera_id": "cam-01"}


@pytest.mark.usefixtures("camera_id")
@pytest.mark.parametrize(
    "face_data",
    [
        {"embedding": "malformed"},
        {"embedding_model_version": "insightface-buffalo_l-1.0"},
        {"snapshot_ref": "snapshots/17.jpg"},
    ],
    ids=lambda face_data: next(iter(face_data)),
)
async def test_any_face_field_is_a_privacy_violation(
    db_session: AsyncSession, face_data: dict[str, Any]
) -> None:
    payload = event()
    payload["participants"][0] |= face_data
    outcome = await ingest(db_session, encode(payload))

    assert outcome.code is FailureCode.BIOMETRIC_NOT_ALLOWED
    assert (await only_audit_entry(db_session)).action == "event.privacy_violation"
    dead_letter = await only_dead_letter(db_session)
    assert dead_letter.raw_payload is not None
    assert not face_data.keys() & dead_letter.raw_payload["participants"][0].keys()


@pytest.mark.usefixtures("camera_id")
@pytest.mark.parametrize(
    ("changes", "expected"),
    [
        pytest.param(
            {"embedding": [math.nan]},
            FailureCode.BIOMETRIC_NOT_ALLOWED,
            id="NaN in the embedding",
        ),
        pytest.param(
            {"embedding": [0.1], "confidence": math.nan},
            FailureCode.INVALID_CONFIDENCE,
            id="NaN next to it",
        ),
    ],
)
async def test_face_data_with_nan_is_audited(
    db_session: AsyncSession, changes: dict[str, Any], expected: FailureCode
) -> None:
    payload = event()
    payload["participants"][0] |= changes
    outcome = await ingest(db_session, encode(payload))

    assert outcome.code is expected
    audit = await only_audit_entry(db_session)
    assert audit.object_id == EVENT_ID
    assert audit.access_context == {"topic": TOPIC, "camera_id": "cam-01"}


@pytest.mark.usefixtures("camera_id")
async def test_rejection_without_face_data_is_not_audited(
    db_session: AsyncSession,
) -> None:
    # Only the privacy violation is audited, not every rejection.
    payload = event(model_version="x" * 129)
    payload["participants"][0] |= {"embedding": None, "snapshot_ref": None}
    outcome = await ingest(db_session, encode(payload))

    assert outcome.code is FailureCode.INVALID_FIELD
    assert await count(db_session, AuditEntry) == 0


@pytest.mark.usefixtures("camera_id")
async def test_null_face_fields_are_accepted(db_session: AsyncSession) -> None:
    payload = event()
    payload["participants"][0] |= {"embedding": None, "snapshot_ref": None}
    outcome = await ingest(db_session, encode(payload))
    assert outcome.status is IngestStatus.ACCEPTED


@pytest.mark.usefixtures("camera_id")
async def test_dead_letter_with_nan_is_written(db_session: AsyncSession) -> None:
    payload = event()
    payload["participants"][0]["confidence"] = float("nan")
    outcome = await ingest(db_session, encode(payload))

    assert outcome.status is IngestStatus.REJECTED
    assert (await only_dead_letter(db_session)).raw_payload is None


@pytest.mark.usefixtures("camera_id")
@pytest.mark.parametrize(
    ("changes", "expected"),
    [
        pytest.param(
            {"confidence": 1.5}, FailureCode.INVALID_CONFIDENCE, id="V13 before V14"
        ),
        pytest.param(
            {"event_type": "suspicion"},
            FailureCode.INVALID_STAGE_TYPE_PAIR,
            id="V7 before V14",
        ),
    ],
)
async def test_privacy_violation_is_audited_whatever_the_reported_code(
    db_session: AsyncSession, changes: dict[str, Any], expected: FailureCode
) -> None:
    # The embedding was sent even if another fault is reported.
    payload = event_with_embedding() | changes
    outcome = await ingest(db_session, encode(payload))

    assert outcome.code is expected
    assert (await only_audit_entry(db_session)).action == "event.privacy_violation"


@pytest.mark.usefixtures("camera_id")
async def test_privacy_violation_keeps_ids_when_payload_is_not_stored(
    db_session: AsyncSession,
) -> None:
    # U+0000 in an ignored field: the dead letter keeps only the hash, but
    # the audit entry still names the event and camera.
    payload = event_with_embedding() | {
        "event_id": EVENT_ID.upper(),
        "note": "a\x00b",
    }
    await ingest(db_session, encode(payload))

    assert (await only_dead_letter(db_session)).raw_payload is None
    audit = await only_audit_entry(db_session)
    assert audit.object_id == EVENT_ID  # canonical form
    assert audit.access_context["camera_id"] == "cam-01"


@pytest.mark.usefixtures("camera_id")
@pytest.mark.parametrize(
    "camera",
    [
        pytest.param("cam\x00", id="U+0000"),
        pytest.param("cam\ud800", id="lone surrogate"),
        pytest.param("c" * 200, id="too long"),
        pytest.param(7, id="not a string"),
    ],
)
async def test_privacy_violation_drops_unstorable_camera_id(
    db_session: AsyncSession, camera: object
) -> None:
    # The audit insert must not fail either, or the message comes back forever.
    payload = event_with_embedding() | {"camera_id": camera}
    outcome = await ingest(db_session, encode(payload))

    assert outcome.status is IngestStatus.REJECTED
    assert (await only_audit_entry(db_session)).access_context["camera_id"] is None


@pytest.mark.usefixtures("camera_id")
async def test_returned_result_is_a_copy(db_session: AsyncSession) -> None:
    # The JSONB column does not track in-place changes, so changing the
    # returned dict must not reach the stored row.
    raw = encode(event())
    first = await ingest(db_session, raw)
    assert first.result is not None
    first.result["incident_id"] = "x"

    second = await ingest(db_session, raw)
    assert second.result == {"outcome": "accepted"}
    assert second.result is not None
    second.result["incident_id"] = "y"
    stored = await db_session.get(ProcessedEvent, uuid.UUID(EVENT_ID))
    assert stored is not None
    assert stored.result == {"outcome": "accepted"}
