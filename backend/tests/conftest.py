from collections.abc import AsyncIterator, Iterator
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy.ext.asyncio import AsyncSession
from testcontainers.community.postgres import PostgresContainer

from app.config import Settings, get_settings
from app.db.session import build_engine

REQUIRED_TEST_ENV = {
    "DB_HOST": "localhost",
    "DB_PORT": "5432",
    "DB_USER": "test",
    "DB_NAME": "test",
    "DB_PASSWORD": "test",
    "MQTT_HOST": "localhost",
    "MQTT_PORT": "1883",
    "MQTT_USERNAME": "test",
    "MQTT_PASSWORD": "test",
    "SESSION_SIGNING_KEY": "test-session-key-0123456789abcdef",
    "CLIP_LINK_SIGNING_KEY": "test-clip-link-key-0123456789abcdef",
}


@pytest.fixture(autouse=True)
def _test_settings(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Provide deterministic settings without loading a local .env file.

    Every Settings variable is removed from the environment first, so values
    left in the shell (e.g. after `source .env`) cannot leak into tests.
    """
    monkeypatch.setitem(Settings.model_config, "env_file", None)

    for field_name in Settings.model_fields:
        monkeypatch.delenv(field_name.upper(), raising=False)

    for name, value in REQUIRED_TEST_ENV.items():
        monkeypatch.setenv(name, value)

    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture(scope="session")
def postgres_container() -> Iterator[PostgresContainer]:
    with PostgresContainer(image="pgvector/pgvector:pg17") as container:
        yield container


@pytest.fixture
def db_settings(
    postgres_container: PostgresContainer, monkeypatch: pytest.MonkeyPatch
) -> Settings:
    monkeypatch.setenv("DB_HOST", postgres_container.get_container_host_ip())
    monkeypatch.setenv("DB_PORT", str(postgres_container.get_exposed_port(5432)))
    monkeypatch.setenv("DB_USER", postgres_container.username)
    monkeypatch.setenv("DB_PASSWORD", postgres_container.password)
    monkeypatch.setenv("DB_NAME", postgres_container.dbname)
    get_settings.cache_clear()
    return get_settings()


@pytest.fixture
def migrated_db(db_settings: Settings) -> Settings:
    config = Config(Path(__file__).resolve().parents[1] / "alembic.ini")
    command.upgrade(config, "head")
    return db_settings


@pytest.fixture
async def db_session(migrated_db: Settings) -> AsyncIterator[AsyncSession]:
    """A session whose changes are rolled back after the test.

    The whole test runs inside one outer transaction. The session joins it
    through a savepoint, so even code that calls session.commit() only
    commits the savepoint, and the final rollback undoes everything.
    Tests therefore never see each other's rows.
    """
    engine = build_engine(migrated_db)
    async with engine.connect() as connection:
        transaction = await connection.begin()
        session = AsyncSession(
            bind=connection,
            join_transaction_mode="create_savepoint",
            expire_on_commit=False,
        )
        try:
            yield session
        finally:
            await session.close()
            await transaction.rollback()
    await engine.dispose()
