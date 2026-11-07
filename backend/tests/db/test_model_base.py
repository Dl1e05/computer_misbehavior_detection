import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm.exc import StaleDataError

from app.db.models import User
from app.domain.incident import Role


async def make_saved_user(db_session: AsyncSession) -> User:
    user = User(login="alice", password_hash="not-a-real-hash", role=Role.OPERATOR)
    db_session.add(user)
    await db_session.flush()
    return user


async def test_version_increments_on_every_update(db_session: AsyncSession) -> None:
    user = await make_saved_user(db_session)
    assert user.version == 1

    user.active = False
    await db_session.flush()
    assert user.version == 2

    user.active = True
    await db_session.flush()
    assert user.version == 3


async def test_concurrent_change_is_detected(db_session: AsyncSession) -> None:
    # Optimistic locking (SRS §12.1): someone else updates the row after we
    # read it, so our update must fail instead of overwriting their change.
    user = await make_saved_user(db_session)
    await db_session.execute(
        text("UPDATE users SET version = version + 1 WHERE id = :id"), {"id": user.id}
    )

    user.active = False
    with pytest.raises(StaleDataError):
        await db_session.flush()


async def test_updated_at_is_readable_after_update(db_session: AsyncSession) -> None:
    # Without eager_defaults this attribute is expired after the UPDATE and
    # reading it raises MissingGreenlet under AsyncSession.
    user = await make_saved_user(db_session)
    user.active = False
    await db_session.flush()

    assert user.updated_at.tzinfo is not None
