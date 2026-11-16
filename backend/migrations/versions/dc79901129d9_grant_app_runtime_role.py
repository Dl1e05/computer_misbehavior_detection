"""grant app_runtime role

The app connects as a login that is a member of app_runtime, never as the
schema owner (an owner ignores GRANT/REVOKE). The login itself, with its
password, is created outside migrations: by the compose init script in dev,
by the deployment procedure otherwise (SRS D3).

Revision ID: dc79901129d9
Revises: a484753808bb
Create Date: 2026-11-16 21:14:47.624592

"""

from collections.abc import Sequence

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "dc79901129d9"
down_revision: str | Sequence[str] | None = "a484753808bb"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    # Roles belong to the whole server, not to one database, so the role may
    # already exist (init script, another database on the same server).
    op.execute(
        """
        DO $$
        BEGIN
            IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'app_runtime') THEN
                CREATE ROLE app_runtime NOLOGIN;
            END IF;
        END
        $$
        """
    )
    op.execute("GRANT USAGE ON SCHEMA public TO app_runtime")
    op.execute(
        "GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public "
        "TO app_runtime"
    )
    op.execute("GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO app_runtime")

    # Append-only audit (BE-FR-75, I8): INSERT and SELECT only.
    op.execute("REVOKE UPDATE, DELETE ON audit_entries FROM app_runtime")
    # Only migrations may change the recorded schema version.
    op.execute("REVOKE ALL ON alembic_version FROM app_runtime")

    # Tables and sequences created later by the migration user get the same
    # rights automatically. A later table that needs narrower rights (like
    # audit_entries) must REVOKE them in its own migration.
    op.execute(
        "ALTER DEFAULT PRIVILEGES IN SCHEMA public "
        "GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO app_runtime"
    )
    op.execute(
        "ALTER DEFAULT PRIVILEGES IN SCHEMA public "
        "GRANT USAGE, SELECT ON SEQUENCES TO app_runtime"
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.execute(
        "ALTER DEFAULT PRIVILEGES IN SCHEMA public "
        "REVOKE USAGE, SELECT ON SEQUENCES FROM app_runtime"
    )
    op.execute(
        "ALTER DEFAULT PRIVILEGES IN SCHEMA public "
        "REVOKE SELECT, INSERT, UPDATE, DELETE ON TABLES FROM app_runtime"
    )
    op.execute("REVOKE ALL ON ALL SEQUENCES IN SCHEMA public FROM app_runtime")
    op.execute("REVOKE ALL ON ALL TABLES IN SCHEMA public FROM app_runtime")
    op.execute("REVOKE USAGE ON SCHEMA public FROM app_runtime")
    # The role is not dropped: it belongs to the server, and logins created
    # outside migrations are members of it.
