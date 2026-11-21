import uuid
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator
from datetime import UTC, datetime
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy.ext.asyncio import AsyncSession
from testcontainers.community.postgres import PostgresContainer

from app.config import Settings, get_settings
from app.db.models import Camera, Incident, Zone
from app.db.session import build_engine
from app.domain.camera import CameraSourceType
from app.domain.incident import IncidentStatus, IncidentType, Severity

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


# The limited login the app connects with.
APP_DB_USER = "app_test"
APP_DB_PASSWORD = "app_test_password"
INIT_SCRIPT = Path(__file__).resolve().parents[1] / "postgres/init/10-app-login.sh"


@pytest.fixture(scope="session")
def postgres_container() -> Iterator[PostgresContainer]:
    with PostgresContainer(image="pgvector/pgvector:pg17") as container:
        # The same script dev uses (postgres/init), so the two cannot drift.
        # POSTGRES_USER/DB/PASSWORD are already in the container environment.
        exit_code, output = container.get_wrapped_container().exec_run(
            ["sh", "-c", INIT_SCRIPT.read_text()],
            environment={
                "APP_DB_USER": APP_DB_USER,
                "APP_DB_PASSWORD": APP_DB_PASSWORD,
            },
        )
        assert exit_code == 0, output.decode()
        yield container


@pytest.fixture
def db_settings(
    postgres_container: PostgresContainer, monkeypatch: pytest.MonkeyPatch
) -> Settings:
    monkeypatch.setenv("DB_HOST", postgres_container.get_container_host_ip())
    monkeypatch.setenv("DB_PORT", str(postgres_container.get_exposed_port(5432)))
    # Tests use the app's limited rights; only migrations use the owner.
    monkeypatch.setenv("DB_USER", APP_DB_USER)
    monkeypatch.setenv("DB_PASSWORD", APP_DB_PASSWORD)
    monkeypatch.setenv("DB_MIGRATION_USER", postgres_container.username)
    monkeypatch.setenv("DB_MIGRATION_PASSWORD", postgres_container.password)
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


# Parent rows that many DB tests need. Fixtures, not importable helpers: with
# --import-mode=importlib, test modules cannot import each other.


MakeCamera = Callable[[str], Awaitable[Camera]]
MakeZone = Callable[[uuid.UUID], Awaitable[Zone]]


@pytest.fixture
def make_camera(db_session: AsyncSession) -> MakeCamera:
    """Adds a camera with the given code; the one place a test Camera is built."""

    async def make(code: str) -> Camera:
        camera = Camera(
            code=code,
            name=f"Camera {code}",
            source_type=CameraSourceType.RTSP,
            source_ref=f"rtsp://media-server/{code}",
        )
        db_session.add(camera)
        await db_session.flush()
        return camera

    return make


@pytest.fixture
def make_zone(db_session: AsyncSession) -> MakeZone:
    """Adds an active zone to the camera; the one place a test Zone is built."""

    async def make(camera_id: uuid.UUID) -> Zone:
        zone = Zone(
            camera_id=camera_id,
            name="Server room door",
            polygon=[[10, 10], [200, 10], [200, 150], [10, 150]],
            frame_width=1920,
            frame_height=1080,
        )
        db_session.add(zone)
        await db_session.flush()
        return zone

    return make


@pytest.fixture
async def camera_id(make_camera: MakeCamera) -> uuid.UUID:
    return (await make_camera("cam-01")).id


@pytest.fixture
async def zone_id(make_zone: MakeZone, camera_id: uuid.UUID) -> uuid.UUID:
    return (await make_zone(camera_id)).id


@pytest.fixture
async def incident_id(db_session: AsyncSession, camera_id: uuid.UUID) -> uuid.UUID:
    incident = Incident(
        incident_type=IncidentType.FIGHT,
        severity=Severity.CRITICAL,
        status=IncidentStatus.SUSPECTED,
        camera_id=camera_id,
        occurred_at=datetime(2026, 1, 1, tzinfo=UTC),
        created_received_at=datetime(2026, 1, 1, 0, 0, 1, tzinfo=UTC),
    )
    db_session.add(incident)
    await db_session.flush()
    return incident.id
