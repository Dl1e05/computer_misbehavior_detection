"""Incident creation from accepted ML events (SRS §5.3, §6)."""

import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Self

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
from app.domain.incident import (
    IncidentStatus,
    IncidentType,
    TimerKind,
    severity_for,
)
from app.domain.state_machine import Actor, TransitionResult, check_transition
from app.mqtt.events import MlEvent


@dataclass(frozen=True)
class IncidentTimings:
    """The settings an incident's deadlines and clip window are built from."""

    escalation: timedelta
    clip_pre_roll: timedelta
    clip_post_roll: timedelta
    clock_skew_warn: timedelta

    @classmethod
    def from_settings(cls, settings: Settings) -> Self:
        return cls(
            escalation=timedelta(seconds=settings.timer_escalation_seconds),
            clip_pre_roll=timedelta(seconds=settings.clip_pre_roll_seconds),
            clip_post_roll=timedelta(seconds=settings.clip_post_roll_seconds),
            clock_skew_warn=timedelta(seconds=settings.clock_skew_warn_seconds),
        )


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
    allowed = check_transition(
        None, IncidentStatus.NEW, Actor.ML_EVENT, event.incident_type
    )
    if allowed is not TransitionResult.ALLOWED:
        raise ValueError(f"{event.incident_type} cannot be created as NEW")

    incident = Incident(
        incident_type=event.incident_type,
        severity=severity_for(event.incident_type),
        status=IncidentStatus.NEW,
        camera_id=camera.id,
        zone_id=zone.id if zone is not None else None,
        zone_version=event.zone_version if zone is not None else None,
        model_version=event.model_version,
        confidence=event.confidence,
        # Only fire has no participant to carry the box.
        bbox=event.bbox if event.incident_type is IncidentType.FIRE else None,
        occurred_at=event.occurred_at,
        created_received_at=received_at,
    )
    session.add(incident)
    # The database creates the id the rows below need.
    await session.flush()

    start_at, end_at = _clip_window(event.occurred_at, received_at, timings)
    session.add_all(
        [
            *_participants(incident.id, event),
            Timer(
                incident_id=incident.id,
                kind=TimerKind.ESCALATION,
                # BE time: a wrong ML clock must not move the deadline.
                deadline_at=received_at + timings.escalation,
            ),
            Clip(incident_id=incident.id, start_at=start_at, end_at=end_at),
            _created_entry(incident, event),
        ]
    )
    await session.flush()
    return incident


def _clip_window(
    occurred_at: datetime, received_at: datetime, timings: IncidentTimings
) -> tuple[datetime, datetime]:
    # §13.4: with a large skew ML's clock cannot be trusted.
    skew = abs(received_at - occurred_at)
    trigger = occurred_at if skew <= timings.clock_skew_warn else received_at
    return trigger - timings.clip_pre_roll, trigger + timings.clip_post_roll


def _participants(incident_id: uuid.UUID, event: MlEvent) -> list[Participant]:
    return [
        Participant(
            incident_id=incident_id,
            run_id=event.run_id,
            track_id=participant.track_id,
            confidence=participant.confidence,
            bbox=participant.bbox,
            model_version=event.model_version,
        )
        for participant in event.participants
    ]


def _created_entry(incident: Incident, event: MlEvent) -> AuditEntry:
    return AuditEntry(
        actor_id=None,
        actor_role=None,
        action="incident.created",
        object_type=AuditObjectType.INCIDENT,
        object_id=str(incident.id),
        result=AuditResult.SUCCESS,
        access_context={
            "event_id": str(event.event_id),
            "trigger": Actor.ML_EVENT.value,
            "from_status": None,
            "to_status": incident.status.value,
        },
    )
