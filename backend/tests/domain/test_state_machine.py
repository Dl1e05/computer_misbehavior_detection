import itertools

import pytest

from app.domain.event import INCIDENT_TYPES_BY_EVENT_TYPE, EventType
from app.domain.incident import TERMINAL_STATUSES, IncidentStatus, IncidentType, Role
from app.domain.state_machine import (
    TRANSITIONS,
    Action,
    Actor,
    TransitionResult,
    allowed_actions,
    check_transition,
)

Case = tuple[IncidentStatus | None, IncidentStatus, Actor, IncidentType]

NON_FIGHT_TYPES = [t for t in IncidentType if t is not IncidentType.FIGHT]

# T4-T8 from SRS §5.3 have no incident-type condition. T4 can only happen to
# fights in practice (only fights are ever SUSPECTED), but the rule itself
# does not depend on the type.
ANY_TYPE_TRANSITIONS: list[tuple[IncidentStatus, IncidentStatus, Actor]] = [
    (IncidentStatus.SUSPECTED, IncidentStatus.NOT_CONFIRMED, Actor.TIMER),  # T4
    (IncidentStatus.NEW, IncidentStatus.IN_PROGRESS, Actor.OPERATOR),  # T5
    (IncidentStatus.IN_PROGRESS, IncidentStatus.CONFIRMED, Actor.OPERATOR),  # T6
    (IncidentStatus.IN_PROGRESS, IncidentStatus.FALSE_ALARM, Actor.OPERATOR),  # T7
    (IncidentStatus.CONFIRMED, IncidentStatus.CLOSED, Actor.HEAD_OF_SECURITY),  # T8
]

# Written from SRS §5.3, independently of the TRANSITIONS table in the code,
# so a wrong row in the code cannot confirm itself.
ALLOWED: set[Case] = {
    (None, IncidentStatus.SUSPECTED, Actor.ML_EVENT, IncidentType.FIGHT),  # T1
    *((None, IncidentStatus.NEW, Actor.ML_EVENT, t) for t in NON_FIGHT_TYPES),  # T2
    (
        IncidentStatus.SUSPECTED,
        IncidentStatus.NEW,
        Actor.ML_EVENT,
        IncidentType.FIGHT,
    ),  # T3
    *(
        (current, target, actor, incident_type)
        for current, target, actor in ANY_TYPE_TRANSITIONS
        for incident_type in IncidentType
    ),  # T4-T8
}


def test_allowed_list_has_expected_size() -> None:
    # T1: 1 + T2: 3 + T3: 1 + T4-T8: 5 x 4 types
    assert len(ALLOWED) == 25


# (actor, target) pairs the SRS lets an actor reach from some status.
REACHABLE: set[tuple[Actor, IncidentStatus]] = {
    (actor, target) for _, target, actor, _ in ALLOWED
}


def expected_result(case: Case) -> TransitionResult:
    _, target, actor, _ = case
    if (actor, target) not in REACHABLE:
        return TransitionResult.FORBIDDEN
    if case not in ALLOWED:
        return TransitionResult.INVALID_TRANSITION
    return TransitionResult.ALLOWED


def test_every_combination_gets_the_srs_result() -> None:
    # An actor that can never reach the target is FORBIDDEN (403) whatever the
    # current status; otherwise anything outside ALLOWED is INVALID (409).
    froms: list[IncidentStatus | None] = [None, *IncidentStatus]
    for case in itertools.product(froms, IncidentStatus, Actor, IncidentType):
        assert check_transition(*case) == expected_result(case), case


def test_plain_strings_behave_like_enum_members() -> None:
    # Values from JSON payloads, tokens or raw SQL rows arrive as str.
    fight_as_new = check_transition(None, "NEW", "ML_EVENT", "fight")  # type: ignore[arg-type]
    fight_suspected = check_transition(None, "SUSPECTED", "ML_EVENT", "fight")  # type: ignore[arg-type]
    operator_takes = check_transition("NEW", "IN_PROGRESS", "OPERATOR", "fire")  # type: ignore[arg-type]

    assert fight_as_new == TransitionResult.INVALID_TRANSITION
    assert fight_suspected == TransitionResult.ALLOWED
    assert operator_takes == TransitionResult.ALLOWED


def test_human_actor_codes_match_role_codes() -> None:
    assert Actor.OPERATOR.value == Role.OPERATOR.value
    assert Actor.HEAD_OF_SECURITY.value == Role.HEAD_OF_SECURITY.value


def test_terminal_statuses_are_exactly_those_without_outgoing_transitions() -> None:
    with_outgoing = {current for current, _ in TRANSITIONS if current is not None}
    assert set(IncidentStatus) - with_outgoing == TERMINAL_STATUSES


@pytest.mark.parametrize(
    ("current", "target", "actor", "incident_type", "expected"),
    [
        pytest.param(
            IncidentStatus.NEW,
            IncidentStatus.IN_PROGRESS,
            Actor.HEAD_OF_SECURITY,
            IncidentType.FIRE,
            TransitionResult.FORBIDDEN,
            id="T5 by head of security is forbidden",
        ),
        pytest.param(
            IncidentStatus.CONFIRMED,
            IncidentStatus.CLOSED,
            Actor.OPERATOR,
            IncidentType.FALL,
            TransitionResult.FORBIDDEN,
            id="T8 by operator is forbidden",
        ),
        pytest.param(
            IncidentStatus.SUSPECTED,
            IncidentStatus.NOT_CONFIRMED,
            Actor.OPERATOR,
            IncidentType.FIGHT,
            TransitionResult.FORBIDDEN,
            id="T4 by operator is forbidden",
        ),
        pytest.param(
            IncidentStatus.NEW,
            IncidentStatus.CLOSED,
            Actor.HEAD_OF_SECURITY,
            IncidentType.FIRE,
            TransitionResult.INVALID_TRANSITION,
            id="NEW to CLOSED is not a transition",
        ),
        pytest.param(
            IncidentStatus.NEW,
            IncidentStatus.CLOSED,
            Actor.OPERATOR,
            IncidentType.FIRE,
            TransitionResult.FORBIDDEN,
            id="operator closing is forbidden in any status",
        ),
        pytest.param(
            IncidentStatus.IN_PROGRESS,
            IncidentStatus.IN_PROGRESS,
            Actor.HEAD_OF_SECURITY,
            IncidentType.FALL,
            TransitionResult.FORBIDDEN,
            id="head of security taking into work is forbidden in any status",
        ),
        pytest.param(
            IncidentStatus.NEW,
            IncidentStatus.NEW,
            Actor.ML_EVENT,
            IncidentType.FIGHT,
            TransitionResult.INVALID_TRANSITION,
            id="a repeat is not a transition",
        ),
        pytest.param(
            IncidentStatus.NEW,
            IncidentStatus.FALSE_ALARM,
            Actor.OPERATOR,
            IncidentType.FALL,
            TransitionResult.INVALID_TRANSITION,
            id="NEW to FALSE_ALARM skips IN_PROGRESS",
        ),
        pytest.param(
            None,
            IncidentStatus.SUSPECTED,
            Actor.ML_EVENT,
            IncidentType.FIRE,
            TransitionResult.INVALID_TRANSITION,
            id="only fights can be SUSPECTED",
        ),
        pytest.param(
            None,
            IncidentStatus.NEW,
            Actor.ML_EVENT,
            IncidentType.FIGHT,
            TransitionResult.INVALID_TRANSITION,
            id="fight cannot be created as NEW directly",
        ),
        pytest.param(
            IncidentStatus.CLOSED,
            IncidentStatus.NEW,
            Actor.ML_EVENT,
            IncidentType.FIRE,
            TransitionResult.INVALID_TRANSITION,
            id="closed incident cannot be reopened",
        ),
    ],
)
def test_refusal_reason(
    current: IncidentStatus | None,
    target: IncidentStatus,
    actor: Actor,
    incident_type: IncidentType,
    expected: TransitionResult,
) -> None:
    assert check_transition(current, target, actor, incident_type) is expected


@pytest.mark.parametrize(
    ("role", "status", "expected"),
    [
        pytest.param(
            Role.OPERATOR,
            IncidentStatus.NEW,
            {Action.TAKE_INTO_WORK},
            id="operator on NEW can take into work",
        ),
        pytest.param(
            Role.OPERATOR,
            IncidentStatus.IN_PROGRESS,
            {Action.CONFIRM, Action.REJECT},
            id="operator on IN_PROGRESS can confirm or reject",
        ),
        pytest.param(
            Role.OPERATOR,
            IncidentStatus.CONFIRMED,
            set(),
            id="operator cannot close",
        ),
        pytest.param(
            Role.HEAD_OF_SECURITY,
            IncidentStatus.CONFIRMED,
            {Action.CLOSE},
            id="head of security on CONFIRMED can close",
        ),
        pytest.param(
            Role.HEAD_OF_SECURITY,
            IncidentStatus.NEW,
            set(),
            id="head of security does not take into work (OQ-03)",
        ),
        pytest.param(
            Role.OPERATOR,
            IncidentStatus.SUSPECTED,
            set(),
            id="no user action on a suspicion",
        ),
    ],
)
def test_allowed_actions(
    role: Role, status: IncidentStatus, expected: set[Action]
) -> None:
    # User actions (T5-T8) apply to every incident type.
    for incident_type in IncidentType:
        assert allowed_actions(role, status, incident_type) == expected, incident_type


def test_administrator_has_no_actions() -> None:
    for status, incident_type in itertools.product(IncidentStatus, IncidentType):
        assert allowed_actions(Role.ADMINISTRATOR, status, incident_type) == set()


def test_terminal_statuses_have_no_actions() -> None:
    for role, status, incident_type in itertools.product(
        Role, TERMINAL_STATUSES, IncidentType
    ):
        assert allowed_actions(role, status, incident_type) == set()


def test_detection_types_match_the_transition_table() -> None:
    # V7 and T2 list the same types. A type allowed only by V7 would fail at
    # creation, so its message would come back forever.
    rule = TRANSITIONS[(None, IncidentStatus.NEW)]
    assert INCIDENT_TYPES_BY_EVENT_TYPE[EventType.DETECTION] == rule.incident_types
