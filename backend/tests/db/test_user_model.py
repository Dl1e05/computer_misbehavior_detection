import uuid

import pytest
from sqlalchemy import func, select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import User
from app.domain.incident import Role


def make_user(login: str = "alice", role: Role = Role.OPERATOR) -> User:
    return User(login=login, password_hash="not-a-real-hash", role=role)


async def test_user_gets_database_defaults(db_session: AsyncSession) -> None:
    user = make_user()
    db_session.add(user)
    await db_session.flush()
    await db_session.refresh(user)

    assert isinstance(user.id, uuid.UUID)
    assert user.active is True
    assert user.version == 1
    assert user.created_at.tzinfo is not None
    assert user.updated_at.tzinfo is not None


async def test_role_is_stored_as_srs_code_and_read_back_as_enum(
    db_session: AsyncSession,
) -> None:
    db_session.add(make_user(role=Role.HEAD_OF_SECURITY))
    await db_session.flush()

    raw = await db_session.scalar(text("SELECT role FROM users"))
    assert raw == "HEAD_OF_SECURITY"

    user = await db_session.scalar(select(User))
    assert user is not None
    assert user.role is Role.HEAD_OF_SECURITY


async def test_login_must_be_unique(db_session: AsyncSession) -> None:
    db_session.add(make_user(login="alice"))
    await db_session.flush()

    db_session.add(make_user(login="alice"))
    with pytest.raises(IntegrityError, match="uq_users_login"):
        await db_session.flush()


async def test_unknown_role_is_rejected_by_check_constraint(
    db_session: AsyncSession,
) -> None:
    # Raw SQL on purpose: the ORM would already refuse an unknown enum value,
    # this test proves the database itself refuses it too (SRS §12.1).
    with pytest.raises(IntegrityError, match="ck_users_role"):
        await db_session.execute(
            text(
                "INSERT INTO users (login, password_hash, role) "
                "VALUES ('mallory', 'x', 'SUPERUSER')"
            )
        )


async def test_each_test_starts_with_an_empty_table(db_session: AsyncSession) -> None:
    # Other tests in this file insert "alice"; if their rows leaked, this
    # count would not be zero.
    count = await db_session.scalar(select(func.count()).select_from(User))
    assert count == 0
