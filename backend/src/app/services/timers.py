"""Firing due timers: T4 and the 30 s escalation (SRS §6.5, BE-FR-24..28)."""

import logging
import uuid
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.db.errors import describe, is_database_unavailable
from app.db.models import AuditEntry, Escalation, Incident, Timer
from app.domain.audit import AuditObjectType, AuditResult
from app.domain.incident import IncidentStatus, TimerKind, TimerState
from app.domain.state_machine import Actor, TransitionResult, check_transition

logger = logging.getLogger(__name__)

BATCH = 100


_EXPECTED = {
    TimerKind.SUSPICION_TIMEOUT: IncidentStatus.SUSPECTED,
    TimerKind.ESCALATION: IncidentStatus.NEW,
}


async def fire_due_timers(
    session_factory: async_sessionmaker[AsyncSession], now: datetime
) -> int:
    async with session_factory() as session:
        due = (
            await session.execute(
                select(Timer.id, Timer.incident_id)
                .where(Timer.state == TimerState.PENDING, Timer.deadline_at <= now)
                .order_by(Timer.deadline_at)
                .limit(BATCH)
            )
        ).all()

    fired = 0
    for timer_id, incident_id in due:
        try:
            if await _fire_one(session_factory, timer_id, incident_id, now):
                fired += 1
        except Exception as error:
            if is_database_unavailable(error):
                raise
            logger.error("timer %s failed (%s)", timer_id, describe(error))
    return fired


async def _fire_one(
    session_factory: async_sessionmaker[AsyncSession],
    timer_id: uuid.UUID,
    incident_id: uuid.UUID,
    now: datetime,
) -> bool:
    async with session_factory() as session:
        incident = await session.scalar(
            select(Incident)
            .where(Incident.id == incident_id)
            .with_for_update(skip_locked=True)
        )
        if incident is None:
            return False
        timer = await session.scalar(
            select(Timer)
            .where(Timer.id == timer_id, Timer.state == TimerState.PENDING)
            .with_for_update()
        )
        if timer is None:
            return False
        if incident.status is not _EXPECTED[timer.kind]:
            timer.state = TimerState.CANCELLED
            await session.commit()
            return False
        if timer.kind is TimerKind.SUSPICION_TIMEOUT:
            allowed = check_transition(
                IncidentStatus.SUSPECTED,
                IncidentStatus.NOT_CONFIRMED,
                Actor.TIMER,
                incident.incident_type,
            )
            if allowed is not TransitionResult.ALLOWED:
                raise ValueError(f"{incident.incident_type} cannot time out")
            incident.status = IncidentStatus.NOT_CONFIRMED
            session.add(
                _timer_entry(
                    incident,
                    timer,
                    action="incident.transition",
                    context={
                        "from_status": IncidentStatus.SUSPECTED.value,
                        "to_status": IncidentStatus.NOT_CONFIRMED.value,
                    },
                )
            )
        else:
            # Escalation: no status change, the incident stays NEW (SRS §5.3)
            session.add(Escalation(incident_id=incident.id, level=1, raised_at=now))
            session.add(
                _timer_entry(
                    incident,
                    timer,
                    action="incident.escalated",
                    context={
                        "level": 1,
                        "severity": incident.severity.value,
                    },
                )
            )
        timer.state = TimerState.FIRED
        timer.fired_at = now
        await session.commit()
        return True


def _timer_entry(
    incident: Incident,
    timer: Timer,
    *,
    action: str,
    context: dict[str, object],
) -> AuditEntry:
    return AuditEntry(
        actor_id=None,
        actor_role=None,
        action=action,
        object_type=AuditObjectType.INCIDENT,
        object_id=str(incident.id),
        result=AuditResult.SUCCESS,
        access_context={
            "timer_id": str(timer.id),
            "trigger": Actor.TIMER.value,
            **context,
        },
    )
