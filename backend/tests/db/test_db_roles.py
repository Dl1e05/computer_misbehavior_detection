"""The app's database rights (BE-FR-75, SRS D3, test T-AU-02).

db_session connects as the limited app login, like the running app does.
"""

import pytest
from sqlalchemy import text
from sqlalchemy.exc import DBAPIError
from sqlalchemy.ext.asyncio import AsyncSession

INSERT_AUDIT_ENTRY = text(
    "INSERT INTO audit_entries (action, object_type, result) "
    "VALUES ('incident.transition', 'incident', 'SUCCESS')"
)


async def test_app_does_not_own_the_schema(db_session: AsyncSession) -> None:
    # An owner ignores GRANT/REVOKE, so every check below would be void.
    current_user = await db_session.scalar(text("SELECT current_user"))
    owner = await db_session.scalar(
        text("SELECT tableowner FROM pg_tables WHERE tablename = 'audit_entries'")
    )
    is_superuser = await db_session.scalar(
        text("SELECT rolsuper FROM pg_roles WHERE rolname = current_user")
    )

    assert current_user != owner
    assert is_superuser is False


async def test_app_can_append_to_audit_and_read_it(db_session: AsyncSession) -> None:
    await db_session.execute(INSERT_AUDIT_ENTRY)
    count = await db_session.scalar(text("SELECT count(*) FROM audit_entries"))
    assert count == 1


@pytest.mark.parametrize(
    "statement",
    [
        pytest.param("UPDATE audit_entries SET reason = 'edited'", id="update"),
        pytest.param("DELETE FROM audit_entries", id="delete"),
        pytest.param("TRUNCATE audit_entries", id="truncate"),
    ],
)
async def test_app_cannot_change_audit(
    db_session: AsyncSession, statement: str
) -> None:
    # T-AU-02: the database itself refuses, whatever the application code does.
    await db_session.execute(INSERT_AUDIT_ENTRY)
    with pytest.raises(DBAPIError, match="permission denied for table audit_entries"):
        await db_session.execute(text(statement))


async def test_app_can_change_operational_tables(db_session: AsyncSession) -> None:
    # Statements on empty tables still check privileges.
    await db_session.execute(text("UPDATE incidents SET measures = 'x'"))
    await db_session.execute(text("DELETE FROM notifications"))


async def test_app_cannot_change_migration_version(db_session: AsyncSession) -> None:
    with pytest.raises(DBAPIError, match="permission denied for table alembic_vers"):
        await db_session.execute(text("UPDATE alembic_version SET version_num = 'x'"))


@pytest.mark.parametrize(
    ("statement", "error"),
    [
        pytest.param(
            "CREATE TABLE sneaky (id int)",
            "permission denied for schema public",
            id="create table",
        ),
        pytest.param(
            "DROP TABLE audit_entries", "must be owner of table", id="drop audit"
        ),
        pytest.param(
            "ALTER TABLE audit_entries DISABLE TRIGGER ALL",
            "must be owner of table",
            id="alter audit",
        ),
    ],
)
async def test_app_cannot_change_the_schema(
    db_session: AsyncSession, statement: str, error: str
) -> None:
    with pytest.raises(DBAPIError, match=error):
        await db_session.execute(text(statement))


async def test_future_tables_get_app_rights_by_default(
    db_session: AsyncSession,
) -> None:
    # ALTER DEFAULT PRIVILEGES: tables added by later migrations are usable
    # by the app without a new GRANT. 'r' = tables, 'S' = sequences.
    granted = await db_session.scalars(
        text(
            "SELECT defaclobjtype::text FROM pg_default_acl, "
            "aclexplode(defaclacl) AS acl "
            "WHERE acl.grantee = 'app_runtime'::regrole"
        )
    )
    assert set(granted) == {"r", "S"}
