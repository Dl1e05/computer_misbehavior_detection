import uuid

import pytest
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import AuditEntry
from app.domain.audit import AuditObjectType, AuditResult
from app.domain.incident import Role


def make_entry(**changes: object) -> AuditEntry:
    entry = AuditEntry(
        actor_id=uuid.uuid4(),
        actor_role=Role.OPERATOR,
        action="incident.transition",
        object_type=AuditObjectType.INCIDENT,
        object_id=str(uuid.uuid4()),
        result=AuditResult.SUCCESS,
    )
    for name, value in changes.items():
        setattr(entry, name, value)
    return entry


async def test_entry_gets_database_defaults(db_session: AsyncSession) -> None:
    entry = make_entry()
    db_session.add(entry)
    await db_session.flush()
    await db_session.refresh(entry)

    assert isinstance(entry.id, int)
    assert entry.access_context == {}
    assert entry.occurred_at.tzinfo is not None
    assert entry.reason is None


async def test_ids_increase_in_insert_order(db_session: AsyncSession) -> None:
    entries = [make_entry() for _ in range(3)]
    for entry in entries:
        db_session.add(entry)
        await db_session.flush()

    ids = [entry.id for entry in entries]
    assert ids == sorted(ids)
    assert len(set(ids)) == 3


async def test_system_action_has_no_actor(db_session: AsyncSession) -> None:
    db_session.add(
        make_entry(
            actor_id=None,
            actor_role=None,
            action="incident.escalated",
            access_context={"event_id": str(uuid.uuid4())},
        )
    )
    await db_session.flush()


async def test_actor_may_be_an_unknown_user(db_session: AsyncSession) -> None:
    # No FK to users: an entry is never refused because of another table.
    db_session.add(make_entry(actor_id=uuid.uuid4()))
    await db_session.flush()


@pytest.mark.parametrize(
    "changes",
    [
        pytest.param({"actor_role": None}, id="actor without role"),
        pytest.param({"actor_id": None}, id="role without actor"),
    ],
)
async def test_actor_fields_go_together(
    db_session: AsyncSession, changes: dict[str, object]
) -> None:
    db_session.add(make_entry(**changes))
    with pytest.raises(IntegrityError, match="ck_audit_entries_actor_fields_together"):
        await db_session.flush()


async def test_denied_entry_keeps_reason(db_session: AsyncSession) -> None:
    entry = make_entry(
        actor_role=Role.ADMINISTRATOR,
        action="auth.denied",
        object_type=AuditObjectType.IDENTIFICATION,
        result=AuditResult.DENIED,
        reason="role may not view identification data",
    )
    db_session.add(entry)
    await db_session.flush()
    await db_session.refresh(entry)

    assert entry.result is AuditResult.DENIED
    assert entry.reason == "role may not view identification data"
