import uuid
from datetime import UTC, datetime

import pytest
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Escalation, User
from app.domain.incident import Role

ACKNOWLEDGED_AT = datetime(2026, 1, 1, 0, 1, tzinfo=UTC)


async def add_head_of_security(db_session: AsyncSession) -> uuid.UUID:
    user = User(
        login="head", password_hash="not-a-real-hash", role=Role.HEAD_OF_SECURITY
    )
    db_session.add(user)
    await db_session.flush()
    return user.id


def make_escalation(incident_id: uuid.UUID, **changes: object) -> Escalation:
    escalation = Escalation(incident_id=incident_id, level=1)
    for name, value in changes.items():
        setattr(escalation, name, value)
    return escalation


async def test_escalation_gets_database_defaults(
    db_session: AsyncSession, incident_id: uuid.UUID
) -> None:
    escalation = make_escalation(incident_id)
    db_session.add(escalation)
    await db_session.flush()
    await db_session.refresh(escalation)

    assert escalation.raised_at.tzinfo is not None
    assert escalation.acknowledged_by is None
    assert escalation.acknowledged_at is None
    assert escalation.version == 1


async def test_level_is_raised_once_per_incident(
    db_session: AsyncSession, incident_id: uuid.UUID
) -> None:
    db_session.add(make_escalation(incident_id, level=1))
    await db_session.flush()

    db_session.add(make_escalation(incident_id, level=1))
    with pytest.raises(IntegrityError, match="uq_escalations_incident_id_level"):
        await db_session.flush()


async def test_next_level_is_accepted(
    db_session: AsyncSession, incident_id: uuid.UUID
) -> None:
    db_session.add_all(
        [make_escalation(incident_id, level=1), make_escalation(incident_id, level=2)]
    )
    await db_session.flush()


async def test_level_must_be_positive(
    db_session: AsyncSession, incident_id: uuid.UUID
) -> None:
    db_session.add(make_escalation(incident_id, level=0))
    with pytest.raises(IntegrityError, match="ck_escalations_level_positive"):
        await db_session.flush()


async def test_acknowledgement_with_both_fields_is_accepted(
    db_session: AsyncSession, incident_id: uuid.UUID
) -> None:
    escalation = make_escalation(incident_id)
    db_session.add(escalation)
    await db_session.flush()

    escalation.acknowledged_by = await add_head_of_security(db_session)
    escalation.acknowledged_at = ACKNOWLEDGED_AT
    await db_session.flush()

    assert escalation.version == 2


@pytest.mark.parametrize(
    "missing", [pytest.param("acknowledged_at"), pytest.param("acknowledged_by")]
)
async def test_acknowledgement_needs_both_fields(
    db_session: AsyncSession, incident_id: uuid.UUID, missing: str
) -> None:
    fields: dict[str, object] = {
        "acknowledged_by": await add_head_of_security(db_session),
        "acknowledged_at": ACKNOWLEDGED_AT,
    }
    del fields[missing]

    db_session.add(make_escalation(incident_id, **fields))
    with pytest.raises(IntegrityError, match="ck_escalations_ack_fields_together"):
        await db_session.flush()


async def test_user_who_acknowledged_cannot_be_deleted(
    db_session: AsyncSession, incident_id: uuid.UUID
) -> None:
    # Escalation history must keep who acknowledged it; users are
    # deactivated, not deleted.
    user_id = await add_head_of_security(db_session)
    db_session.add(
        make_escalation(
            incident_id, acknowledged_by=user_id, acknowledged_at=ACKNOWLEDGED_AT
        )
    )
    await db_session.flush()

    with pytest.raises(IntegrityError, match="fk_escalations_acknowledged_by_users"):
        await db_session.execute(
            text("DELETE FROM users WHERE id = :id"), {"id": user_id}
        )


async def test_escalations_are_removed_with_their_incident(
    db_session: AsyncSession, incident_id: uuid.UUID
) -> None:
    db_session.add(make_escalation(incident_id))
    await db_session.flush()

    await db_session.execute(
        text("DELETE FROM incidents WHERE id = :id"), {"id": incident_id}
    )

    remaining = await db_session.scalar(
        text("SELECT count(*) FROM escalations WHERE incident_id = :id"),
        {"id": incident_id},
    )
    assert remaining == 0
