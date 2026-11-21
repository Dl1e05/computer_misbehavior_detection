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
from app.mqtt.events import EventLimits, MlEvent, Rejection, parse_event

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

    # V1..V7, V9, V11 (first half), V12..V15.
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
            return IngestOutcome(IngestStatus.DUPLICATE, result=stored.result)
        return await reject(
            Rejection(
                FailureCode.EVENT_ID_CONFLICT,
                "event_id: already processed with another payload",
            )
        )

    # V8. Known deviation from the V order: parse_event has already applied
    # V9 and V11..V15, so an unknown camera next to such a fault is reported
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
    # 10b adds the incident and its id to the result. If two copies of one
    # event race, the second INSERT hits the primary key and its transaction
    # rolls back; unacknowledged, the copy comes back and is a DUPLICATE.
    result: dict[str, Any] = {"outcome": "accepted"}
    session.add(
        ProcessedEvent(
            event_id=event.event_id, payload_hash=payload_hash, result=result
        )
    )
    await session.flush()
    return IngestOutcome(IngestStatus.ACCEPTED, result=result)


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
    payload = storable_payload(raw)
    session.add(
        DeadLetter(
            received_at=limits.received_at,
            topic=topic,
            failure_code=rejection.code.value,
            payload_hash=payload_hash,
            raw_payload=payload,
        )
    )
    if rejection.code is FailureCode.BIOMETRIC_NOT_ALLOWED:
        # §7.5: a privacy violation record in the Audit Log.
        session.add(_privacy_violation(payload, topic))
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


def _privacy_violation(payload: dict[str, Any] | None, topic: str) -> AuditEntry:
    # The message passed every field check, so event_id and camera_id are
    # valid strings; payload is None only if it could not be stored.
    payload = payload or {}
    event_id = payload.get("event_id")
    camera_id = payload.get("camera_id")
    return AuditEntry(
        actor_id=None,
        actor_role=None,
        action="event.privacy_violation",
        object_type=AuditObjectType.EVENT,
        object_id=event_id if isinstance(event_id, str) else None,
        result=AuditResult.DENIED,
        reason=FailureCode.BIOMETRIC_NOT_ALLOWED.value,
        access_context={
            "topic": topic,
            "camera_id": camera_id if isinstance(camera_id, str) else None,
        },
    )


def storable_payload(raw: bytes) -> dict[str, Any] | None:
    """The payload as it may be kept in a dead letter, or None.

    A dead letter must never fail to save: that would roll back the
    transaction and the message would be redelivered forever. So anything
    Postgres JSONB refuses, or that is too large, becomes None and only the
    payload hash is kept. Embeddings are removed at any depth (SRS §12.6, §14).
    """
    try:
        loaded = json.loads(raw, parse_constant=_refuse_constant)
        if not isinstance(loaded, dict):
            return None
        cleaned: dict[str, Any] = _without_embeddings(loaded)
        # Lone surrogates (valid in Python's json, refused by JSONB) fail to
        # encode as UTF-8 (UnicodeEncodeError is a ValueError).
        encoded = json.dumps(cleaned, ensure_ascii=False).encode()
    except (ValueError, RecursionError):
        # Not UTF-8, not JSON, NaN/Infinity, lone surrogates, or nested too
        # deeply for any of the three steps.
        return None
    # json.dumps always escapes U+0000, and JSONB refuses that escape. A
    # literal "\\u0000" text also matches; dropping it as well is harmless.
    if b"\\u0000" in encoded or len(encoded) > DEAD_LETTER_PAYLOAD_MAX_BYTES:
        return None
    return cleaned


def _refuse_constant(name: str) -> Any:
    raise ValueError(f"{name} is not valid JSON")


def _without_embeddings(value: Any) -> Any:
    # The payload is unvalidated, so embeddings may be anywhere.
    if isinstance(value, dict):
        return {
            key: _without_embeddings(item)
            for key, item in value.items()
            if key != "embedding"
        }
    if isinstance(value, list):
        return [_without_embeddings(item) for item in value]
    return value
