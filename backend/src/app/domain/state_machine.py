from enum import StrEnum

from app.domain.incident import IncidentStatus, IncidentType, Role


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


class Action(StrEnum):
    """Status changes a logged-in user can request."""

    TAKE_INTO_WORK = "take_into_work"  # T5
    CONFIRM = "confirm"  # T6
    REJECT = "reject"  # T7
    CLOSE = "close"  # T8


ACTION_TARGET: dict[Action, IncidentStatus] = {
    Action.TAKE_INTO_WORK: IncidentStatus.IN_PROGRESS,
    Action.CONFIRM: IncidentStatus.CONFIRMED,
    Action.REJECT: IncidentStatus.FALSE_ALARM,
    Action.CLOSE: IncidentStatus.CLOSED,
}

# The Administrator is deliberately absent: they never change incident statuses.
ROLE_ACTOR: dict[Role, Actor] = {
    Role.OPERATOR: Actor.OPERATOR,
    Role.HEAD_OF_SECURITY: Actor.HEAD_OF_SECURITY,
}


def allowed_actions(
    role: Role,
    status: IncidentStatus,
    incident_type: IncidentType,
) -> frozenset[Action]:
    """Actions the role may perform in this status.

    Derived from check_transition so the rules live in one place. The result
    tells the frontend which buttons to show; every request is still checked
    again with check_transition.
    """
    actor = ROLE_ACTOR.get(role)
    if actor is None:
        return frozenset()

    return frozenset(
        action
        for action, target in ACTION_TARGET.items()
        if check_transition(status, target, actor, incident_type)
        is TransitionResult.ALLOWED
    )
