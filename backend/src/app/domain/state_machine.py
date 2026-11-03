from enum import StrEnum

from app.domain.incident import IncidentStatus, IncidentType


class Actor(StrEnum):
    ML_EVENT = "ml_event"
    TIMER = "timer"
    OPERATOR = "operator"
    HEAD_OF_SECURITY = "head_of_security"


class TransitionResult(StrEnum):
    ALLOWED = "allowed"
    INVALID_TRANSITION = "invalid_transition"
    FORBIDDEN = "forbidden"


TRANSITIONS: dict[tuple[IncidentStatus | None, IncidentStatus], Actor] = {
    (None, IncidentStatus.SUSPECTED): Actor.ML_EVENT,  # T1
    (None, IncidentStatus.NEW): Actor.ML_EVENT,  # T2
    (IncidentStatus.SUSPECTED, IncidentStatus.NEW): Actor.ML_EVENT,  # T3
    (IncidentStatus.SUSPECTED, IncidentStatus.NOT_CONFIRMED): Actor.TIMER,  # T4
    (IncidentStatus.NEW, IncidentStatus.IN_PROGRESS): Actor.OPERATOR,  # T5
    (IncidentStatus.IN_PROGRESS, IncidentStatus.CONFIRMED): Actor.OPERATOR,  # T6
    (IncidentStatus.IN_PROGRESS, IncidentStatus.FALSE_ALARM): Actor.OPERATOR,  # T7
    (IncidentStatus.CONFIRMED, IncidentStatus.CLOSED): Actor.HEAD_OF_SECURITY,  # T8
}


FIGHT_ONLY = frozenset(
    {
        (None, IncidentStatus.SUSPECTED),  # T1
        (IncidentStatus.SUSPECTED, IncidentStatus.NEW),  # T3
    }
)


FIGHT_FORBIDDEN = frozenset(
    {
        (None, IncidentStatus.NEW),  # T2
    }
)


def check_transition(
    current: IncidentStatus | None,
    target: IncidentStatus,
    actor: Actor,
    incident_type: IncidentType,
) -> TransitionResult:
    pair = (current, target)
    allowed_actor = TRANSITIONS.get(pair)

    if allowed_actor is None:
        return TransitionResult.INVALID_TRANSITION

    if actor is not allowed_actor:
        return TransitionResult.FORBIDDEN

    if pair in FIGHT_ONLY and incident_type is not IncidentType.FIGHT:
        return TransitionResult.INVALID_TRANSITION

    if pair in FIGHT_FORBIDDEN and incident_type is IncidentType.FIGHT:
        return TransitionResult.INVALID_TRANSITION

    return TransitionResult.ALLOWED
