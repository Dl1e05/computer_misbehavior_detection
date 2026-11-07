from sqlalchemy import text
from testcontainers.community.postgres import PostgresContainer

from app.config import Settings
from app.db.session import build_engine


def test_settings_point_at_container(
    postgres_container: PostgresContainer, db_settings: Settings
) -> None:
    assert db_settings.db_port == postgres_container.get_exposed_port(5432)


async def test_pgvector_extension_installed(migrated_db: Settings) -> None:
    engine = build_engine(migrated_db)
    try:
        async with engine.connect() as connection:
            result = await connection.execute(
                text("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
            )
            version = result.scalar_one_or_none()
    finally:
        await engine.dispose()

    assert version is not None
