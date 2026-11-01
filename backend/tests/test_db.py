from testcontainers.community.postgres import PostgresContainer

from app.config import Settings


def test_settings_point_at_container(
    postgres_container: PostgresContainer, db_settings: Settings
) -> None:
    assert db_settings.db_port == postgres_container.get_exposed_port(5432)
