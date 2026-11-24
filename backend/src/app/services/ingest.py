"""Event ingestion: one MQTT message in, one outcome out (SRS §6.4, §7.4, §7.5).

The caller owns the transaction: ingest_event only flushes. The MQTT handler
commits and only then acknowledges the message, so a crash before the commit
leaves the message unacknowledged and the broker delivers it again (§7.1).
"""

import hashlib
import json
import logging
import uuid
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import AuditEntry, Camera, DeadLetter, ProcessedEvent, Zone
from app.domain.audit import AuditObjectType, AuditResult
from app.domain.event import FailureCode
from app.mqtt.events import (
    FACE_FIELDS,
    EventLimits,
    MlEvent,
    Rejection,
    parse_event,
)

logger = logging.getLogger(__name__)

# SRS §7.1: 256 KB per event without embeddings. A larger dead letter keeps
# only its hash.
DEAD_LETTER_PAYLOAD_MAX_BYTES = 256 * 1024


class IngestStatus(StrEnum):
    ACCEPTED = "accepted"
    DUPLICATE = "duplicate"
    REJECTED = "rejected"


@dataclass(frozen=True)
class IngestOutcome:
    status: IngestStatus
    # Set for REJECTED only.
    code: FailureCode | None = None
    # The stored processing result; set for ACCEPTED and DUPLICATE.
    result: dict[str, Any] | None = None


async def ingest_event(
    session: AsyncSession, raw: bytes, *, topic: str, limits: EventLimits
) -> IngestOutcome:
    """Validate and record one event; write nothing that cannot be stored.

    Every path ends in exactly one of: a processed_events row (accepted),
    nothing new (duplicate), or a dead_letters row (rejected).
    """
    payload_hash = hashlib.sha256(raw).hexdigest()

    async def reject(rejection: Rejection) -> IngestOutcome:
        return await _reject(
            session,
            raw,
            rejection,
            topic=topic,
            limits=limits,
            payload_hash=payload_hash,
        )

    # V1..V7, V9, V11 (first half), V12..V14.
    parsed = parse_event(raw, limits)
    if isinstance(parsed, Rejection):
        return await reject(parsed)
    event = parsed

    # V16 runs before the database checks on purpose: a camera disabled after
    # the first delivery must not turn a redelivery into UNKNOWN_CAMERA.
    # "Already processed" wins over everything else.
    stored = await session.get(ProcessedEvent, event.event_id)
    if stored is not None:
        if stored.payload_hash == payload_hash:
            # A copy: the JSONB column does not track in-place changes.
            return IngestOutcome(IngestStatus.DUPLICATE, result=dict(stored.result))
        return await reject(
            Rejection(
                FailureCode.EVENT_ID_CONFLICT,
                "event_id: already processed with another payload",
            )
        )

    # V8. Known deviation from the V order: parse_event has already applied
    # V9 and V11..V14, so an unknown camera next to such a fault is reported
    # as that fault. Handling (§7.5) is the same, only the code differs.
    camera = await session.scalar(select(Camera).where(Camera.code == event.camera_id))
    if camera is None or not camera.enabled:
        return await reject(
            Rejection(FailureCode.UNKNOWN_CAMERA, "camera_id: unknown or disabled")
        )

    # V11, second half. Checked whenever zone_id is present, not only for
    # zone_intrusion, so no incident ever points at another camera's zone.
    if event.zone_id is not None and not await _zone_is_usable(
        session, event.zone_id, camera.id
    ):
        return await reject(
            Rejection(
                FailureCode.INVALID_ZONE,
                "zone_id: unknown, inactive or of another camera",
            )
        )

    return await _accept(session, event, payload_hash)


async def _zone_is_usable(
    session: AsyncSession, zone_id: str, camera_id: uuid.UUID
) -> bool:
    # zone_id is the Zone's UUID, as ML reads it from GET /ml/cameras-zones.
    try:
        zone_uuid = uuid.UUID(zone_id)
    except ValueError:
        return False
    zone = await session.get(Zone, zone_uuid)
    return zone is not None and zone.camera_id == camera_id and zone.active


async def _accept(
    session: AsyncSession, event: MlEvent, payload_hash: str
) -> IngestOutcome:
    # 10b adds the incident and its id to the result. Build the whole result
    # before adding the row: the JSONB column does not track later in-place
    # changes, so the caller gets a copy.
    # If two copies of one event race, the second INSERT hits the primary key
    # and its transaction rolls back; unacknowledged, the copy comes back and
    # is a DUPLICATE.
    result: dict[str, Any] = {"outcome": "accepted"}
    session.add(
        ProcessedEvent(
            event_id=event.event_id, payload_hash=payload_hash, result=result
        )
    )
    await session.flush()
    return IngestOutcome(IngestStatus.ACCEPTED, result=dict(result))


async def _reject(
    session: AsyncSession,
    raw: bytes,
    rejection: Rejection,
    *,
    topic: str,
    limits: EventLimits,
    payload_hash: str,
) -> IngestOutcome:
    # Rejections are not stored in processed_events: a redelivered rejection
    # (lost PUBACK) only adds another dead letter, which is harmless.
    loaded = load_object(raw)
    session.add(
        DeadLetter(
            received_at=limits.received_at,
            topic=topic,
            failure_code=rejection.code.value,
            payload_hash=payload_hash,
            raw_payload=storable_payload(loaded),
        )
    )
    # §7.5: a privacy violation record in the Audit Log. Decided from the
    # message itself, not from the reported code: with another fault in the
    # message the code may be that fault (V order), but face data was still
    # sent.
    if loaded is not None and _has_face_data(loaded):
        session.add(_privacy_violation(loaded, topic))
    await session.flush()
    # detail never contains input values (see Rejection), so it may be logged.
    logger.warning(
        "event rejected: %s (%s) topic=%s hash=%s",
        rejection.code.value,
        rejection.detail,
        topic,
        payload_hash,
    )
    return IngestOutcome(IngestStatus.REJECTED, code=rejection.code)


def _has_face_data(loaded: dict[str, Any]) -> bool:
    # The same condition as V14 in MlEvent, read from the unvalidated JSON.
    participants = loaded.get("participants")
    return isinstance(participants, list) and any(
        isinstance(participant, dict)
        and any(participant.get(field) is not None for field in FACE_FIELDS)
        for participant in participants
    )


def _privacy_violation(loaded: dict[str, Any], topic: str) -> AuditEntry:
    # The message may have failed any check, so its values are untrusted:
    # each one is kept only if Postgres can store it, or the audit insert
    # would fail and the message would be redelivered forever.
    return AuditEntry(
        actor_id=None,
        actor_role=None,
        action="event.privacy_violation",
        object_type=AuditObjectType.EVENT,
        object_id=_canonical_uuid(loaded.get("event_id")),
        result=AuditResult.DENIED,
        reason=FailureCode.BIOMETRIC_NOT_ALLOWED.value,
        access_context={
            "topic": topic,
            "camera_id": _storable_text(loaded.get("camera_id")),
        },
    )


def _canonical_uuid(value: object) -> str | None:
    # Canonical form, so lookups by the event's id find the entry.
    if not isinstance(value, str):
        return None
    try:
        return str(uuid.UUID(value))
    except ValueError:
        return None


# A camera code is short; anything longer is not one.
_MAX_TEXT = 128


def _storable_text(value: object) -> str | None:
    if not isinstance(value, str) or "\x00" in value or len(value) > _MAX_TEXT:
        return None
    try:
        value.encode()
    except UnicodeEncodeError:  # lone surrogate
        return None
    return value


def load_object(raw: bytes) -> dict[str, Any] | None:
    """The message as a JSON object, or None if it is not one."""
    try:
        loaded = json.loads(raw)
    except (ValueError, RecursionError):
        # Not UTF-8, not JSON, an integer with thousands of digits, or nested
        # too deeply. NaN and Infinity are loaded; storable_payload refuses them.
        return None
    return loaded if isinstance(loaded, dict) else None


def storable_payload(loaded: dict[str, Any] | None) -> dict[str, Any] | None:
    """The payload as it may be kept in a dead letter, or None.

    A dead letter must never fail to save: that would roll back the
    transaction and the message would be redelivered forever. So anything
    Postgres JSONB refuses, or that is too large, becomes None and only the
    payload hash is kept. Face data is removed at any depth (SRS §12.6, §14).
    """
    if loaded is None:
        return None
    try:
        cleaned: dict[str, Any] = _without_face_data(loaded)
        # allow_nan=False: JSONB refuses NaN and Infinity, also the inf that an
        # overflowing number such as 1e400 loads as. Lone surrogates fail to
        # encode as UTF-8. Both raise a ValueError.
        encoded = json.dumps(cleaned, ensure_ascii=False, allow_nan=False).encode()
    except (ValueError, RecursionError):
        return None
    # json.dumps always escapes U+0000, and JSONB refuses that escape. A
    # literal "\\u0000" text also matches; dropping it as well is harmless.
    if b"\\u0000" in encoded or len(encoded) > DEAD_LETTER_PAYLOAD_MAX_BYTES:
        return None
    return cleaned


def _without_face_data(value: Any) -> Any:
    # The payload is unvalidated, so face data may be anywhere.
    if isinstance(value, dict):
        return {
            key: _without_face_data(item)
            for key, item in value.items()
            if key not in FACE_FIELDS
        }
    if isinstance(value, list):
        return [_without_face_data(item) for item in value]
    return value
