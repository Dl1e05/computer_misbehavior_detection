"""Event ingestion: one MQTT message in, one outcome out (SRS §6.4, §7.4, §7.5).

The caller owns the transaction: ingest_event only flushes. The MQTT handler
commits and only then acknowledges the message, so a crash before the commit
leaves the message unacknowledged and the broker delivers it again (§7.1).
"""

import hashlib
import json
import logging
import re
import uuid
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import AuditEntry, Camera, DeadLetter, ProcessedEvent, Zone
from app.domain.audit import AuditObjectType, AuditResult
from app.domain.event import EventType, FailureCode
from app.mqtt.events import (
    FACE_FIELDS,
    EventLimits,
    MlEvent,
    Rejection,
    parse_event,
)
from app.services.incidents import (
    IncidentTimings,
    InvalidTransition,
    create_from_detection,
    handle_confirmation,
    handle_suspicion,
)

logger = logging.getLogger(__name__)

# SRS §7.1: 256 KB per event without embeddings. A larger dead letter keeps
# only its hash.
DEAD_LETTER_PAYLOAD_MAX_BYTES = 256 * 1024

# videoanalytics/v1/cameras/{camera_id}/events (SRS §7.1).
_EVENT_TOPIC = re.compile(r"videoanalytics/v1/cameras/([^/]+)/events")


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
    session: AsyncSession,
    raw: bytes,
    *,
    topic: str,
    limits: EventLimits,
    timings: IncidentTimings,
) -> IngestOutcome:
    """Validate and record one event; write nothing that cannot be stored.

    Every path ends in exactly one of: a processed_events row (accepted, with
    its incident), nothing new (duplicate), or a dead_letters row (rejected).
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

    # The topic names the camera as well. If the two differ, one of them is
    # wrong, and the incident could appear at the wrong place.
    if _topic_camera(topic) != event.camera_id:
        return await reject(
            Rejection(FailureCode.UNKNOWN_CAMERA, "camera_id: does not match the topic")
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
    zone: Zone | None = None
    if event.zone_id is not None:
        zone = await _usable_zone(session, event.zone_id, camera.id)
        if zone is None:
            return await reject(
                Rejection(
                    FailureCode.INVALID_ZONE,
                    "zone_id: unknown, inactive or of another camera",
                )
            )

    try:
        return await _accept(
            session,
            event,
            payload_hash=payload_hash,
            camera=camera,
            zone=zone,
            received_at=limits.received_at,
            timings=timings,
        )
    except InvalidTransition:
        # V17. Raised before anything is written, so only the dead letter
        # is saved. A fixed text: detail never holds input values.
        return await reject(
            Rejection(
                FailureCode.INVALID_TRANSITION,
                "event_type: does not fit the incident's status",
            )
        )


def _topic_camera(topic: str) -> str | None:
    match = _EVENT_TOPIC.fullmatch(topic)
    return match.group(1) if match is not None else None


async def _usable_zone(
    session: AsyncSession, zone_id: str, camera_id: uuid.UUID
) -> Zone | None:
    # zone_id is the Zone's UUID, as ML reads it from GET /ml/cameras-zones.
    try:
        zone_uuid = uuid.UUID(zone_id)
    except ValueError:
        return None
    zone = await session.get(Zone, zone_uuid)
    if zone is None or zone.camera_id != camera_id or not zone.active:
        return None
    return zone


async def _accept(
    session: AsyncSession,
    event: MlEvent,
    *,
    payload_hash: str,
    camera: Camera,
    zone: Zone | None,
    received_at: datetime,
    timings: IncidentTimings,
) -> IngestOutcome:
    # T2 for detection; T1 and T3 for fights (OQ-19). Every accepted event
    # belongs to an incident, a new one or one it repeats.
    if event.event_type is EventType.DETECTION:
        handle = create_from_detection
    elif event.event_type is EventType.SUSPICION:
        handle = handle_suspicion
    else:
        handle = handle_confirmation
    incident = await handle(
        session,
        event,
        camera=camera,
        zone=zone,
        received_at=received_at,
        timings=timings,
    )
    # After the incident: a rejected event (InvalidTransition) is not warned
    # about, its dead letter is enough.
    _warn_if_skewed(event, received_at, timings)

    # Build the whole result before adding the row: the JSONB column does not
    # track later in-place changes, so the caller gets a copy.
    # If two copies of one event race, the second INSERT hits the primary key
    # and its transaction rolls back, incident included; unacknowledged, the
    # copy comes back and is a DUPLICATE. Two first events of one fight race
    # the same way on uq_incidents_open_episode; the loser comes back and
    # finds the fight. So the MQTT handler must roll back and not acknowledge
    # on any error.
    result: dict[str, Any] = {
        "outcome": "accepted",
        "incident_id": str(incident.id),
        "event_type": event.event_type.value,
        "run_id": event.run_id,
    }
    session.add(
        ProcessedEvent(
            event_id=event.event_id, payload_hash=payload_hash, result=result
        )
    )
    await session.flush()
    return IngestOutcome(IngestStatus.ACCEPTED, result=dict(result))


def _warn_if_skewed(
    event: MlEvent, received_at: datetime, timings: IncidentTimings
) -> None:
    # V10, §6.6: a warning only, never a rejection.
    if timings.is_skewed(event.occurred_at, received_at):
        logger.warning(
            "clock skew: camera=%s event=%s occurred_at=%s received_at=%s skew=%.3fs",
            event.camera_id,
            event.event_id,
            event.occurred_at.isoformat(),
            received_at.isoformat(),
            (received_at - event.occurred_at).total_seconds(),
        )


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
