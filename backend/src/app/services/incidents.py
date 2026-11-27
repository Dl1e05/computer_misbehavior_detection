"""Incident creation from accepted ML events (SRS §5.3, §6)."""

import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Self

from sqlalchemy import func, select, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings
from app.db.models import (
    AuditEntry,
    Camera,
    Clip,
    Incident,
    Participant,
    Timer,
    Zone,
)
from app.domain.audit import AuditObjectType, AuditResult
from app.domain.event import TYPES_WITHOUT_PARTICIPANTS
from app.domain.incident import (
    TERMINAL_STATUSES,
    IncidentStatus,
    TimerKind,
    TimerState,
    severity_for,
)
from app.domain.state_machine import Actor, TransitionResult, check_transition
from app.mqtt.events import MlEvent


@dataclass(frozen=True)
class IncidentTimings:
    """The settings an incident's deadlines and clip window are built from."""

    suspicion: timedelta
    escalation: timedelta
    clip_pre_roll: timedelta
    clip_post_roll: timedelta
    clock_skew_warn: timedelta

    @classmethod
    def from_settings(cls, settings: Settings) -> Self:
        return cls(
            suspicion=timedelta(seconds=settings.timer_suspicion_seconds),
            escalation=timedelta(seconds=settings.timer_escalation_seconds),
            clip_pre_roll=timedelta(seconds=settings.clip_pre_roll_seconds),
            clip_post_roll=timedelta(seconds=settings.clip_post_roll_seconds),
            clock_skew_warn=timedelta(seconds=settings.clock_skew_warn_seconds),
        )

    def is_skewed(self, occurred_at: datetime, received_at: datetime) -> bool:
        # Either clock may be ahead.
        return abs(received_at - occurred_at) > self.clock_skew_warn


async def create_from_detection(
    session: AsyncSession,
    event: MlEvent,
    *,
    camera: Camera,
    zone: Zone | None,
    received_at: datetime,
    timings: IncidentTimings,
) -> Incident:
    """Create a NEW incident with its participants, timer, clip and audit entry.

    Does not commit: the caller saves everything together with the processed
    event, or nothing at all.
    """
    _require_transition(None, IncidentStatus.NEW, event)

    # Every detection creates an incident for now; attaching repeats (BE-FR-19)
    # needs OQ-04 first.
    incident = Incident(
        incident_type=event.incident_type,
        severity=severity_for(event.incident_type),
        status=IncidentStatus.NEW,
        camera_id=camera.id,
        zone_id=zone.id if zone is not None else None,
        zone_version=event.zone_version if zone is not None else None,
        model_version=event.model_version,
        confidence=event.confidence,
        # Other types always have participants, which carry the boxes.
        bbox=(
            event.bbox if event.incident_type in TYPES_WITHOUT_PARTICIPANTS else None
        ),
        occurred_at=event.occurred_at,
        created_received_at=received_at,
    )
    session.add(incident)
    # The database creates the id the rows below need.
    await session.flush()

    await _save_participants(session, incident.id, event)
    start_at, end_at = _clip_window(event.occurred_at, received_at, timings)
    session.add_all(
        [
            Timer(
                incident_id=incident.id,
                kind=TimerKind.ESCALATION,
                # BE time: a wrong ML clock must not move the deadline.
                deadline_at=received_at + timings.escalation,
            ),
            Clip(incident_id=incident.id, start_at=start_at, end_at=end_at),
            _status_entry(incident, event, from_status=None),
        ]
    )
    await session.flush()
    return incident


class InvalidTransition(Exception):
    """The event does not fit the incident's current status (V17)."""


async def handle_suspicion(
    session: AsyncSession,
    event: MlEvent,
    *,
    camera: Camera,
    zone: Zone | None,
    received_at: datetime,
    timings: IncidentTimings,
) -> Incident:
    """T1, or the open fight this suspicion repeats.

    Raises InvalidTransition for a suspicion that arrives after its fight
    was confirmed (§7.8). Does not commit.
    """
    fight = await _open_fight(session, camera.id, event.episode_id)
    if fight is None:
        return await _create_suspected(
            session,
            event,
            camera=camera,
            zone=zone,
            received_at=received_at,
            timings=timings,
        )
    if fight.status is IncidentStatus.SUSPECTED:
        # A repeated suspicion changes nothing; the timer keeps its deadline.
        return fight
    raise InvalidTransition(f"suspicion for a {fight.status.value} fight")


async def handle_confirmation(
    session: AsyncSession,
    event: MlEvent,
    *,
    camera: Camera,
    zone: Zone | None,
    received_at: datetime,
    timings: IncidentTimings,
) -> Incident:
    """T3; T1 and T3 together if no suspicion is known (OQ-19). Does not commit."""
    fight = await _open_fight(session, camera.id, event.episode_id)
    if fight is None:
        fight = await _create_suspected(
            session,
            event,
            camera=camera,
            zone=zone,
            received_at=received_at,
            timings=timings,
        )
    elif fight.status is not IncidentStatus.SUSPECTED:
        # Already confirmed: a repeat. Attaching its data comes with 10d.
        return fight
    await _confirm(session, fight, event, received_at=received_at, timings=timings)
    return fight


async def _open_fight(
    session: AsyncSession, camera_id: uuid.UUID, episode_id: str | None
) -> Incident | None:
    # "" is no episode either: it must not join two different fights.
    if not episode_id:
        return None
    statement = (
        select(Incident)
        .where(
            Incident.camera_id == camera_id,
            Incident.episode_id == episode_id,
            Incident.status.not_in(TERMINAL_STATUSES),
        )
        # Locked until commit, so the timer worker cannot end the fight while
        # this event changes it. populate_existing: read the locked row, not
        # an older copy the session may hold.
        .with_for_update()
        .execution_options(populate_existing=True)
    )
    # At most one row: uq_incidents_open_episode.
    return await session.scalar(statement)


async def _create_suspected(
    session: AsyncSession,
    event: MlEvent,
    *,
    camera: Camera,
    zone: Zone | None,
    received_at: datetime,
    timings: IncidentTimings,
) -> Incident:
    # T1: no alarm and no clip (T4 needs no archive record).
    _require_transition(None, IncidentStatus.SUSPECTED, event)
    incident = Incident(
        incident_type=event.incident_type,
        severity=severity_for(event.incident_type),
        status=IncidentStatus.SUSPECTED,
        camera_id=camera.id,
        zone_id=zone.id if zone is not None else None,
        zone_version=event.zone_version if zone is not None else None,
        episode_id=event.episode_id or None,
        model_version=event.model_version,
        confidence=event.confidence,
        occurred_at=event.occurred_at,
        created_received_at=received_at,
    )
    session.add(incident)
    await session.flush()

    await _save_participants(session, incident.id, event)
    session.add_all(
        [
            Timer(
                incident_id=incident.id,
                kind=TimerKind.SUSPICION_TIMEOUT,
                deadline_at=received_at + timings.suspicion,
            ),
            _status_entry(incident, event, from_status=None),
        ]
    )
    await session.flush()
    return incident


async def _confirm(
    session: AsyncSession,
    fight: Incident,
    event: MlEvent,
    *,
    received_at: datetime,
    timings: IncidentTimings,
) -> None:
    # T3. Also when the 10 s deadline has passed but the timer worker has not
    # run yet: the fight is still SUSPECTED, and refusing would lose an alarm.
    _require_transition(IncidentStatus.SUSPECTED, IncidentStatus.NEW, event)
    fight.status = IncidentStatus.NEW
    fight.model_version = event.model_version
    if event.confidence is not None:
        fight.confidence = event.confidence

    await session.execute(
        update(Timer)
        .where(
            Timer.incident_id == fight.id,
            Timer.kind == TimerKind.SUSPICION_TIMEOUT,
            Timer.state == TimerState.PENDING,
        )
        .values(state=TimerState.CANCELLED, version=Timer.version + 1)
    )
    await _save_participants(session, fight.id, event)
    # The confirmation is the trigger of the alarm and the clip (§13.4).
    start_at, end_at = _clip_window(event.occurred_at, received_at, timings)
    session.add_all(
        [
            Timer(
                incident_id=fight.id,
                kind=TimerKind.ESCALATION,
                deadline_at=received_at + timings.escalation,
            ),
            Clip(incident_id=fight.id, start_at=start_at, end_at=end_at),
            _status_entry(fight, event, from_status=IncidentStatus.SUSPECTED),
        ]
    )
    await session.flush()


def _require_transition(
    current: IncidentStatus | None, target: IncidentStatus, event: MlEvent
) -> None:
    # A programming error, not a bad message: V7 already admits only the
    # types the transition table allows (see test_state_machine).
    allowed = check_transition(current, target, Actor.ML_EVENT, event.incident_type)
    if allowed is not TransitionResult.ALLOWED:
        raise ValueError(f"{event.incident_type} cannot go to {target.value}")


def _clip_window(
    occurred_at: datetime, received_at: datetime, timings: IncidentTimings
) -> tuple[datetime, datetime]:
    # §13.4: with a large skew ML's clock cannot be trusted.
    skewed = timings.is_skewed(occurred_at, received_at)
    trigger = received_at if skewed else occurred_at
    return trigger - timings.clip_pre_roll, trigger + timings.clip_post_roll


async def _save_participants(
    session: AsyncSession, incident_id: uuid.UUID, event: MlEvent
) -> None:
    # A confirmation usually has the suspicion's tracks again; a plain INSERT
    # would hit the unique key, and the event would come back forever. The
    # row is updated instead, keeping the old box if the new event has none.
    if not event.participants:
        return
    statement = insert(Participant).values(
        [
            {
                "incident_id": incident_id,
                "run_id": event.run_id,
                "track_id": participant.track_id,
                "confidence": participant.confidence,
                "bbox": participant.bbox,
                "model_version": event.model_version,
            }
            for participant in event.participants
        ]
    )
    await session.execute(
        statement.on_conflict_do_update(
            index_elements=["incident_id", "run_id", "track_id"],
            set_={
                "confidence": statement.excluded.confidence,
                "bbox": func.coalesce(statement.excluded.bbox, Participant.bbox),
                # Core does not apply the ORM's onupdate and version counter.
                "updated_at": func.now(),
                "version": Participant.version + 1,
            },
        )
    )


def _status_entry(
    incident: Incident, event: MlEvent, *, from_status: IncidentStatus | None
) -> AuditEntry:
    # Creation (T1, T2) and status changes (T3) have different actions; both
    # record the step, so the history can be rebuilt from the audit log.
    return AuditEntry(
        actor_id=None,
        actor_role=None,
        action="incident.created" if from_status is None else "incident.transition",
        object_type=AuditObjectType.INCIDENT,
        object_id=str(incident.id),
        result=AuditResult.SUCCESS,
        access_context={
            "event_id": str(event.event_id),
            "trigger": Actor.ML_EVENT.value,
            "from_status": from_status.value if from_status is not None else None,
            "to_status": incident.status.value,
        },
    )
