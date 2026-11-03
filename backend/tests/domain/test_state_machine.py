import itertools

import pytest

from app.domain.incident import IncidentStatus, IncidentType
from app.domain.state_machine import Actor, TransitionResult, check_transition

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


def test_only_srs_transitions_are_allowed() -> None:
    froms: list[IncidentStatus | None] = [None, *IncidentStatus]
    for case in itertools.product(froms, IncidentStatus, Actor, IncidentType):
        result = check_transition(*case)
        assert (result is TransitionResult.ALLOWED) == (case in ALLOWED), case


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
