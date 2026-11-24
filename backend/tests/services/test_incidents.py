import json
import uuid
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings, get_settings
from app.db.models import AuditEntry, Camera, Clip, Incident, Participant, Timer, Zone
from app.domain.audit import AuditObjectType, AuditResult
from app.domain.incident import (
    ClipRecordState,
    ClipState,
    IncidentStatus,
    IncidentType,
    Severity,
    TimerKind,
    TimerState,
)
from app.mqtt.events import EventLimits, MlEvent, parse_event
from app.services.incidents import IncidentTimings, create_from_detection

RECEIVED_AT = datetime(2026, 10, 25, 9, 14, 4, tzinfo=UTC)
LIMITS = EventLimits(received_at=RECEIVED_AT, max_future=timedelta(seconds=60))
# All different, so a swapped value is noticed.
TIMINGS = IncidentTimings(
    escalation=timedelta(seconds=30),
    clip_pre_roll=timedelta(seconds=5),
    clip_post_roll=timedelta(seconds=7),
    clock_skew_warn=timedelta(seconds=2),
)
EVENT_ID = "5f0c2a7e-3b9d-4c1a-9e8f-2d7b6a1c0e42"
BOX = [0.1, 0.2, 0.3, 0.4]


def event(**changes: Any) -> MlEvent:
    """A valid detection, as ingest gets it from the parser."""
    payload = {
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
    parsed = parse_event(json.dumps(payload).encode(), LIMITS)
    assert isinstance(parsed, MlEvent), parsed
    return parsed


@pytest.fixture
async def camera(db_session: AsyncSession, camera_id: uuid.UUID) -> Camera:
    return await db_session.get_one(Camera, camera_id)


@pytest.fixture
async def zone(db_session: AsyncSession, zone_id: uuid.UUID) -> Zone:
    return await db_session.get_one(Zone, zone_id)


async def create(
    session: AsyncSession, event: MlEvent, camera: Camera, zone: Zone | None = None
) -> Incident:
    """Creates the incident and reads it back as the database stored it."""
    incident = await create_from_detection(
        session,
        event,
        camera=camera,
        zone=zone,
        received_at=RECEIVED_AT,
        timings=TIMINGS,
    )
    await session.refresh(incident)
    return incident


async def stored[M](session: AsyncSession, model: type[M]) -> Sequence[M]:
    """Every row as the database stored it, not as the session remembers it."""
    statement = select(model).execution_options(populate_existing=True)
    return (await session.scalars(statement)).all()


# The incident


@pytest.mark.parametrize(
    ("incident_type", "severity"),
    [
        (IncidentType.ZONE_INTRUSION, Severity.MEDIUM),
        (IncidentType.FALL, Severity.HIGH),
        (IncidentType.FIRE, Severity.CRITICAL),
    ],
)
async def test_detection_creates_a_new_incident(
    db_session: AsyncSession,
    camera: Camera,
    zone: Zone,
    incident_type: IncidentType,
    severity: Severity,
) -> None:
    detection = event(
        incident_type=incident_type.value, zone_id=str(zone.id), confidence=0.8
    )
    incident = await create(db_session, detection, camera, zone)

    assert incident.incident_type is incident_type
    assert incident.severity is severity
    assert incident.status is IncidentStatus.NEW
    assert incident.clip_state is ClipState.PENDING
    assert incident.camera_id == camera.id
    assert incident.model_version == "fall-1.0"
    assert incident.confidence == 0.8
    assert incident.occurred_at == detection.occurred_at
    assert incident.created_received_at == RECEIVED_AT


async def test_optional_values_may_be_missing(
    db_session: AsyncSession, camera: Camera
) -> None:
    incident = await create(db_session, event(), camera)

    assert incident.confidence is None
    assert incident.bbox is None
    assert incident.zone_id is None
    assert incident.zone_version is None


async def test_zone_and_its_version_are_kept(
    db_session: AsyncSession, camera: Camera, zone: Zone
) -> None:
    # Version 0 is a version, not "no version".
    detection = event(zone_id=str(zone.id), zone_version=0)
    incident = await create(db_session, detection, camera, zone)

    assert incident.zone_id == zone.id
    assert incident.zone_version == 0


async def test_zone_version_without_a_zone_is_not_kept(
    db_session: AsyncSession, camera: Camera
) -> None:
    incident = await create(db_session, event(zone_version=3), camera)
    assert incident.zone_version is None


@pytest.mark.parametrize("event_type", ["suspicion", "confirmation"])
async def test_fight_cannot_be_created_as_new(
    db_session: AsyncSession, camera: Camera, event_type: str
) -> None:
    # A call by mistake must fail and leave nothing behind.
    fight = event(event_type=event_type, incident_type="fight")
    with pytest.raises(ValueError, match="fight"):
        await create(db_session, fight, camera)

    assert await stored(db_session, Incident) == []


# Boxes and participants


async def test_fire_keeps_its_box_and_has_no_participants(
    db_session: AsyncSession, camera: Camera
) -> None:
    fire = event(incident_type="fire", participants=[], bbox=BOX)
    incident = await create(db_session, fire, camera)

    assert incident.bbox == BOX
    assert await stored(db_session, Participant) == []


async def test_event_box_is_kept_only_for_fire(
    db_session: AsyncSession, camera: Camera
) -> None:
    incident = await create(db_session, event(bbox=BOX), camera)
    assert incident.bbox is None


async def test_every_participant_is_saved(
    db_session: AsyncSession, camera: Camera
) -> None:
    detection = event(
        participants=[
            # Track ids start at 0.
            {"track_id": 0, "confidence": 0.9, "bbox": BOX},
            {"track_id": 5, "confidence": 0.5},
        ]
    )
    incident = await create(db_session, detection, camera)

    first, second = sorted(
        await stored(db_session, Participant), key=lambda row: row.track_id
    )
    for participant in (first, second):
        assert participant.incident_id == incident.id
        assert participant.run_id == "run-1"
        assert participant.model_version == "fall-1.0"
        assert participant.snapshot_ref is None
    assert (first.track_id, first.confidence, first.bbox) == (0, 0.9, BOX)
    assert (second.track_id, second.confidence, second.bbox) == (5, 0.5, None)


# Timer and clip


async def test_escalation_timer_counts_from_receipt(
    db_session: AsyncSession, camera: Camera
) -> None:
    # A late event: ML time must not shorten the operator's 30 seconds.
    late = event(occurred_at="2026-10-25T09:00:00Z")
    incident = await create(db_session, late, camera)

    (timer,) = await stored(db_session, Timer)
    assert timer.incident_id == incident.id
    assert timer.kind is TimerKind.ESCALATION
    assert timer.state is TimerState.PENDING
    assert timer.deadline_at == RECEIVED_AT + timedelta(seconds=30)


AT_LIMIT = RECEIVED_AT - timedelta(seconds=2)
ML_AHEAD_AT_LIMIT = RECEIVED_AT + timedelta(seconds=2)


@pytest.mark.parametrize(
    ("occurred_at", "trigger"),
    [
        pytest.param(RECEIVED_AT, RECEIVED_AT, id="no skew"),
        pytest.param(AT_LIMIT, AT_LIMIT, id="exactly at the limit"),
        pytest.param(
            AT_LIMIT - timedelta(microseconds=1), RECEIVED_AT, id="above the limit"
        ),
        pytest.param(
            ML_AHEAD_AT_LIMIT, ML_AHEAD_AT_LIMIT, id="ML ahead, exactly at the limit"
        ),
        pytest.param(
            ML_AHEAD_AT_LIMIT + timedelta(microseconds=1),
            RECEIVED_AT,
            id="ML ahead, above the limit",
        ),
        pytest.param(datetime(1970, 1, 1, tzinfo=UTC), RECEIVED_AT, id="ML clock lost"),
    ],
)
async def test_clip_window(
    db_session: AsyncSession, camera: Camera, occurred_at: datetime, trigger: datetime
) -> None:
    detection = event(occurred_at=occurred_at.isoformat())
    incident = await create(db_session, detection, camera)

    (clip,) = await stored(db_session, Clip)
    assert clip.incident_id == incident.id
    assert clip.state is ClipRecordState.PENDING
    assert clip.start_at == trigger - timedelta(seconds=5)
    assert clip.end_at == trigger + timedelta(seconds=7)


# Audit


async def test_creation_is_audited_once(
    db_session: AsyncSession, camera: Camera
) -> None:
    incident = await create(db_session, event(), camera)

    (entry,) = await stored(db_session, AuditEntry)
    assert entry.action == "incident.created"
    assert entry.object_type is AuditObjectType.INCIDENT
    assert entry.object_id == str(incident.id)
    assert entry.result is AuditResult.SUCCESS
    # The system, not a user.
    assert entry.actor_id is None
    assert entry.actor_role is None
    assert entry.access_context == {
        "event_id": EVENT_ID,
        "trigger": "ML_EVENT",
        "from_status": None,
        "to_status": "NEW",
    }


# Timings


def test_timings_come_from_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TIMER_ESCALATION_SECONDS", "11")
    monkeypatch.setenv("CLIP_PRE_ROLL_SECONDS", "12")
    monkeypatch.setenv("CLIP_POST_ROLL_SECONDS", "13")
    monkeypatch.setenv("CLOCK_SKEW_WARN_SECONDS", "14")
    get_settings.cache_clear()
    settings: Settings = get_settings()

    assert IncidentTimings.from_settings(settings) == IncidentTimings(
        escalation=timedelta(seconds=11),
        clip_pre_roll=timedelta(seconds=12),
        clip_post_roll=timedelta(seconds=13),
        clock_skew_warn=timedelta(seconds=14),
    )
