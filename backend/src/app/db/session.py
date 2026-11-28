import asyncio
import logging
from collections.abc import AsyncIterator

from fastapi import Request
from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.config import Settings
from app.db.errors import describe, is_database_unavailable

logger = logging.getLogger(__name__)


def build_engine(settings: Settings) -> AsyncEngine:
    # pre_ping checks a pooled connection before use, so the app recovers
    # after the database container is recreated.
    return create_async_engine(settings.database_url, pool_pre_ping=True)


class UnsafeDatabaseRoleError(RuntimeError):
    """The app's database login could bypass the append-only audit."""


async def check_app_role_is_limited(engine: AsyncEngine) -> None:
    """Refuse to run as a superuser or as (a member of) the audit's owner.

    Both ignore GRANT/REVOKE, so the append-only audit (BE-FR-75) would not
    hold. Asked from the database itself, so a renamed login cannot slip
    through the way a config name check could.
    """
    async with engine.connect() as connection:
        row = (
            await connection.execute(
                text(
                    "SELECT r.rolsuper, "
                    "pg_has_role(current_user, t.tableowner, 'USAGE') "
                    "FROM pg_roles AS r, pg_tables AS t "
                    "WHERE r.rolname = current_user "
                    "AND t.schemaname = 'public' AND t.tablename = 'audit_entries'"
                )
            )
        ).one_or_none()
    if row is None:
        raise UnsafeDatabaseRoleError(
            "audit_entries not found: run 'alembic upgrade head' first"
        )
    is_superuser, owns_audit = row
    if is_superuser or owns_audit:
        raise UnsafeDatabaseRoleError(
            "DB_USER is a superuser or owns the schema; the app must connect "
            "as the limited app_runtime login"
        )


async def wait_for_database(
    engine: AsyncEngine,
    *,
    first_delay: float = 1.0,
    max_delay: float = 30.0,
) -> None:
    """Checks the database login, waiting while the database is unreachable.

    The worker may start before Postgres is ready (compose, reboot). An
    unsafe login still stops it at once.
    """
    delay = first_delay
    while True:
        try:
            await check_app_role_is_limited(engine)
        except Exception as error:
            if not is_database_unavailable(error):
                raise
            logger.warning(
                "database not reachable (%s), retry in %.0fs", describe(error), delay
            )
            await asyncio.sleep(delay)
            delay = min(delay * 2, max_delay)
        else:
            return


def create_session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(engine, expire_on_commit=False)


async def get_session(request: Request) -> AsyncIterator[AsyncSession]:
    session_factory = request.app.state.session_factory
    async with session_factory() as session:
        yield session
