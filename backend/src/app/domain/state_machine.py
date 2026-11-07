from enum import StrEnum
from typing import NamedTuple

from app.domain.incident import IncidentStatus, IncidentType, Role


class Actor(StrEnum):
    """Who triggers a status change (SRS §5.3, "Triggered by").

    Human actors use the same codes as Role, so Audit entries match the stored
    role codes.
    """

    ML_EVENT = "ML_EVENT"
    TIMER = "TIMER"
    OPERATOR = Role.OPERATOR.value
    HEAD_OF_SECURITY = Role.HEAD_OF_SECURITY.value


class TransitionResult(StrEnum):
    ALLOWED = "allowed"
    INVALID_TRANSITION = "invalid_transition"  # -> 409
    FORBIDDEN = "forbidden"  # -> 403


class TransitionRule(NamedTuple):
    actor: Actor
    incident_types: frozenset[IncidentType]


ALL_TYPES = frozenset(IncidentType)
FIGHT_ONLY = frozenset({IncidentType.FIGHT})

# SRS §5.3. Every row lists its allowed incident types explicitly (an
# allow-list), so a new incident type is not admitted anywhere by accident.
TRANSITIONS: dict[tuple[IncidentStatus | None, IncidentStatus], TransitionRule] = {
    (None, IncidentStatus.SUSPECTED): TransitionRule(Actor.ML_EVENT, FIGHT_ONLY),  # T1
    (None, IncidentStatus.NEW): TransitionRule(
        Actor.ML_EVENT,
        frozenset({IncidentType.ZONE_INTRUSION, IncidentType.FALL, IncidentType.FIRE}),
    ),  # T2
    (IncidentStatus.SUSPECTED, IncidentStatus.NEW): TransitionRule(
        Actor.ML_EVENT, FIGHT_ONLY
    ),  # T3
    (IncidentStatus.SUSPECTED, IncidentStatus.NOT_CONFIRMED): TransitionRule(
        Actor.TIMER, ALL_TYPES
    ),  # T4
    (IncidentStatus.NEW, IncidentStatus.IN_PROGRESS): TransitionRule(
        Actor.OPERATOR, ALL_TYPES
    ),  # T5
    (IncidentStatus.IN_PROGRESS, IncidentStatus.CONFIRMED): TransitionRule(
        Actor.OPERATOR, ALL_TYPES
    ),  # T6
    (IncidentStatus.IN_PROGRESS, IncidentStatus.FALSE_ALARM): TransitionRule(
        Actor.OPERATOR, ALL_TYPES
    ),  # T7
    (IncidentStatus.CONFIRMED, IncidentStatus.CLOSED): TransitionRule(
        Actor.HEAD_OF_SECURITY, ALL_TYPES
    ),  # T8
}

# Target statuses each actor can ever reach, from any status.
REACHABLE_BY: dict[Actor, frozenset[IncidentStatus]] = {
    actor: frozenset(
        target for (_, target), rule in TRANSITIONS.items() if rule.actor == actor
    )
    for actor in Actor
}


def check_transition(
    current: IncidentStatus | None,
    target: IncidentStatus,
    actor: Actor,
    incident_type: IncidentType,
) -> TransitionResult:
    """Decide whether `actor` may move an incident from `current` to `target`.

    - FORBIDDEN (403): the actor can never reach `target`, whatever the current
      status. Checked first, so the answer does not reveal the incident's
      status to an actor without the permission (SRS §5.5, §10).
    - INVALID_TRANSITION (409): the actor may use this action in general, but
      not from `current`, or not for this incident type.

    Comparisons use ==, so plain strings equal to the enum values (e.g. from a
    JSON payload) behave exactly like the enum members.

    Repeats are not transitions: a repeated `detection`/`confirmation` for an
    open incident (SRS §6.3) must be detected by the caller before this is
    called; (NEW, NEW) is correctly INVALID_TRANSITION here.
    """
    if target not in REACHABLE_BY.get(actor, frozenset()):
        return TransitionResult.FORBIDDEN

    rule = TRANSITIONS.get((current, target))
    if rule is None or rule.actor != actor or incident_type not in rule.incident_types:
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
        == TransitionResult.ALLOWED
    )
