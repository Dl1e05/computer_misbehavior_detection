import json
import uuid
from collections.abc import Awaitable, Callable, Sequence
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from sqlalchemy import event as sa_event
from sqlalchemy import select, update
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
from app.services.incidents import (
    IncidentTimings,
    InvalidTransition,
    create_from_detection,
    handle_confirmation,
    handle_suspicion,
)

RECEIVED_AT = datetime(2026, 10, 25, 9, 14, 4, tzinfo=UTC)
LIMITS = EventLimits(received_at=RECEIVED_AT, max_future=timedelta(seconds=60))
# All different, so a swapped value is noticed.
TIMINGS = IncidentTimings(
    suspicion=timedelta(seconds=10),
    escalation=timedelta(seconds=30),
    clip_pre_roll=timedelta(seconds=5),
    clip_post_roll=timedelta(seconds=7),
    clock_skew_warn=timedelta(seconds=2),
)
EVENT_ID = "5f0c2a7e-3b9d-4c1a-9e8f-2d7b6a1c0e42"
BOX = [0.1, 0.2, 0.3, 0.4]

# The make_camera fixture (tests/conftest.py); test modules cannot import
# conftest types.
MakeCamera = Callable[[str], Awaitable[Camera]]


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
        suspicion=timedelta(seconds=10),
        escalation=timedelta(seconds=11),
        clip_pre_roll=timedelta(seconds=12),
        clip_post_roll=timedelta(seconds=13),
        clock_skew_warn=timedelta(seconds=14),
    )


# Fights (T1, T3; OQ-19)

CONFIRMED_AT = RECEIVED_AT + timedelta(seconds=4)


def fight(event_type: str, **changes: Any) -> MlEvent:
    """A fight suspicion or confirmation of episode-1."""
    return event(
        **{
            "event_id": str(uuid.uuid4()),
            "event_type": event_type,
            "incident_type": "fight",
            "episode_id": "episode-1",
            "model_version": f"fight-{event_type}-1.0",
            "confidence": 0.5,
            "participants": [{"track_id": 17, "confidence": 0.5, "bbox": BOX}],
        }
        | changes
    )


async def handle(
    session: AsyncSession,
    event: MlEvent,
    camera: Camera,
    received_at: datetime = RECEIVED_AT,
) -> Incident:
    """Handles a fight event as ingest would."""
    handler = (
        handle_suspicion if event.event_type == "suspicion" else handle_confirmation
    )
    return await handler(
        session,
        event,
        camera=camera,
        zone=None,
        received_at=received_at,
        timings=TIMINGS,
    )


def timer_states(timers: Sequence[Timer]) -> dict[TimerKind, TimerState]:
    return {timer.kind: timer.state for timer in timers}


async def history(session: AsyncSession) -> list[tuple[str, str | None, str]]:
    """The audit entries as (action, from status, to status), oldest first."""
    entries = sorted(await stored(session, AuditEntry), key=lambda entry: entry.id)
    return [
        (
            entry.action,
            entry.access_context["from_status"],
            entry.access_context["to_status"],
        )
        for entry in entries
    ]


async def test_suspicion_creates_a_suspected_fight(
    db_session: AsyncSession, camera: Camera
) -> None:
    incident = await handle(db_session, fight("suspicion"), camera)
    await db_session.refresh(incident)

    assert incident.status is IncidentStatus.SUSPECTED
    assert incident.severity is Severity.CRITICAL
    assert incident.episode_id == "episode-1"
    assert incident.created_received_at == RECEIVED_AT
    assert incident.clip_state is ClipState.NOT_REQUIRED
    (timer,) = await stored(db_session, Timer)
    assert timer.kind is TimerKind.SUSPICION_TIMEOUT
    assert timer.state is TimerState.PENDING
    assert timer.deadline_at == RECEIVED_AT + timedelta(seconds=10)
    # No alarm yet, so no clip (T4 needs no archive record).
    assert await stored(db_session, Clip) == []
    assert await history(db_session) == [("incident.created", None, "SUSPECTED")]


async def test_confirmation_makes_the_fight_new(
    db_session: AsyncSession, camera: Camera
) -> None:
    suspected = await handle(db_session, fight("suspicion"), camera)
    confirmation = fight(
        "confirmation", confidence=0.9, occurred_at="2026-10-25T09:14:07Z"
    )
    confirmed = await handle(db_session, confirmation, camera, CONFIRMED_AT)

    assert confirmed.id == suspected.id
    (incident,) = await stored(db_session, Incident)
    assert incident.status is IncidentStatus.NEW
    assert incident.clip_state is ClipState.PENDING
    # The confirmation's data; the fight still started at the suspicion.
    assert incident.confidence == 0.9
    assert incident.model_version == "fight-confirmation-1.0"
    assert incident.created_received_at == RECEIVED_AT

    timers = {timer.kind: timer for timer in await stored(db_session, Timer)}
    assert timers[TimerKind.SUSPICION_TIMEOUT].state is TimerState.CANCELLED
    escalation = timers[TimerKind.ESCALATION]
    assert escalation.state is TimerState.PENDING
    assert escalation.deadline_at == CONFIRMED_AT + timedelta(seconds=30)

    # The confirmation triggers the clip (§13.4).
    (clip,) = await stored(db_session, Clip)
    assert clip.start_at == confirmation.occurred_at - timedelta(seconds=5)
    assert await history(db_session) == [
        ("incident.created", None, "SUSPECTED"),
        ("incident.transition", "SUSPECTED", "NEW"),
    ]


async def test_confirmation_without_suspicion_records_both_steps(
    db_session: AsyncSession, camera: Camera
) -> None:
    # OQ-19 default: T1 and T3 in one transaction (T-INC-07).
    incident = await handle(db_session, fight("confirmation"), camera)

    assert incident.status is IncidentStatus.NEW
    assert timer_states(await stored(db_session, Timer)) == {
        TimerKind.SUSPICION_TIMEOUT: TimerState.CANCELLED,
        TimerKind.ESCALATION: TimerState.PENDING,
    }
    assert len(await stored(db_session, Clip)) == 1
    assert await history(db_session) == [
        ("incident.created", None, "SUSPECTED"),
        ("incident.transition", "SUSPECTED", "NEW"),
    ]


async def test_repeated_suspicion_changes_nothing(
    db_session: AsyncSession, camera: Camera
) -> None:
    first = await handle(db_session, fight("suspicion"), camera)
    later = RECEIVED_AT + timedelta(seconds=5)
    repeat = await handle(db_session, fight("suspicion"), camera, later)

    assert repeat.id == first.id
    # The timer keeps its deadline: a repeat must not delay T4.
    (timer,) = await stored(db_session, Timer)
    assert timer.deadline_at == RECEIVED_AT + timedelta(seconds=10)
    assert len(await history(db_session)) == 1


async def test_repeated_confirmation_changes_nothing(
    db_session: AsyncSession, camera: Camera
) -> None:
    first = await handle(db_session, fight("confirmation"), camera)
    repeat = await handle(db_session, fight("confirmation"), camera, CONFIRMED_AT)

    assert repeat.id == first.id
    assert len(await stored(db_session, Incident)) == 1
    assert len(await stored(db_session, Clip)) == 1
    assert len(await history(db_session)) == 2


async def test_suspicion_after_confirmation_is_an_invalid_transition(
    db_session: AsyncSession, camera: Camera
) -> None:
    # Reordered delivery (§7.8, T-EV-05).
    await handle(db_session, fight("confirmation"), camera)

    with pytest.raises(InvalidTransition):
        await handle(db_session, fight("suspicion"), camera, CONFIRMED_AT)

    (incident,) = await stored(db_session, Incident)
    assert incident.status is IncidentStatus.NEW


@pytest.mark.parametrize("status", [IncidentStatus.FALSE_ALARM, IncidentStatus.CLOSED])
async def test_suspicion_after_the_fight_ended_is_an_invalid_transition(
    db_session: AsyncSession, camera: Camera, status: IncidentStatus
) -> None:
    # From the ML outbox backlog, after the operator already handled the fight.
    ended = await handle(db_session, fight("confirmation"), camera)
    ended.status = status
    ended.measures = "Guards sent"
    await db_session.flush()

    with pytest.raises(InvalidTransition):
        await handle(db_session, fight("suspicion"), camera, CONFIRMED_AT)

    # A new confirmation still raises the alarm.
    again = await handle(db_session, fight("confirmation"), camera, CONFIRMED_AT)
    assert again.id != ended.id
    assert again.status is IncidentStatus.NEW


async def test_status_is_read_from_the_database(
    db_session: AsyncSession, camera: Camera
) -> None:
    # The session still holds the incident as SUSPECTED; the locked row
    # must be read again, or the late suspicion would pass as a repeat.
    incident = await handle(db_session, fight("suspicion"), camera)
    await db_session.execute(
        update(Incident)
        .where(Incident.id == incident.id)
        .values(status=IncidentStatus.NEW, version=Incident.version + 1)
        .execution_options(synchronize_session=False)
    )

    with pytest.raises(InvalidTransition):
        await handle(db_session, fight("suspicion"), camera, CONFIRMED_AT)


async def test_open_fight_is_locked(db_session: AsyncSession, camera: Camera) -> None:
    # The timer worker must not end the fight while a confirmation changes
    # it. One test session cannot show a lock, so the SQL is checked.
    await handle(db_session, fight("suspicion"), camera)
    statements: list[str] = []

    def capture(*args: Any) -> None:
        statements.append(args[2])  # (connection, cursor, statement, ...)

    engine = (await db_session.connection()).sync_engine
    sa_event.listen(engine, "before_cursor_execute", capture)
    try:
        await handle(db_session, fight("confirmation"), camera, CONFIRMED_AT)
    finally:
        sa_event.remove(engine, "before_cursor_execute", capture)

    (lookup,) = [sql for sql in statements if "WHERE incidents.camera_id" in sql]
    assert lookup.rstrip().endswith("FOR UPDATE")


@pytest.mark.parametrize("episode_id", [None, ""])
async def test_fight_without_an_episode_never_matches(
    db_session: AsyncSession, camera: Camera, episode_id: str | None
) -> None:
    # The confirmation still raises the alarm; the suspicion only times out.
    await handle(db_session, fight("suspicion", episode_id=episode_id), camera)
    await handle(db_session, fight("confirmation", episode_id=episode_id), camera)

    incidents = await stored(db_session, Incident)
    assert sorted(incident.status for incident in incidents) == [
        IncidentStatus.NEW,
        IncidentStatus.SUSPECTED,
    ]
    assert all(incident.episode_id is None for incident in incidents)


async def test_ended_fight_does_not_match(
    db_session: AsyncSession, camera: Camera
) -> None:
    # Agreed with ML: after NOT_CONFIRMED the same episode starts a new fight.
    ended = await handle(db_session, fight("suspicion"), camera)
    ended.status = IncidentStatus.NOT_CONFIRMED
    await db_session.flush()

    new = await handle(db_session, fight("suspicion"), camera, CONFIRMED_AT)

    assert new.id != ended.id
    assert new.status is IncidentStatus.SUSPECTED


async def test_same_episode_on_another_camera_is_another_fight(
    db_session: AsyncSession, camera: Camera, make_camera: MakeCamera
) -> None:
    other = await make_camera("cam-02")
    first = await handle(db_session, fight("suspicion"), camera)
    second = await handle(db_session, fight("suspicion", camera_id="cam-02"), other)

    assert second.id != first.id


async def test_confirmation_updates_the_suspicion_participants(
    db_session: AsyncSession, camera: Camera
) -> None:
    # Same track again: one row, new confidence, old box kept when the
    # confirmation has none. A plain insert would hit the unique key.
    await handle(db_session, fight("suspicion"), camera)
    confirmation = fight(
        "confirmation",
        participants=[
            {"track_id": 17, "confidence": 0.95},
            {"track_id": 0, "confidence": 0.7},
        ],
    )
    await handle(db_session, confirmation, camera, CONFIRMED_AT)

    participants = sorted(
        await stored(db_session, Participant), key=lambda row: row.track_id
    )
    assert [
        (row.track_id, row.confidence, row.bbox, row.version) for row in participants
    ] == [(0, 0.7, None, 1), (17, 0.95, BOX, 2)]
    # The confidence belongs to the model that gave it.
    assert {row.model_version for row in participants} == {"fight-confirmation-1.0"}
