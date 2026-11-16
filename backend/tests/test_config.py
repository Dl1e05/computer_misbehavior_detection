import pytest
from pydantic import SecretStr

from app.config import Settings


def make_settings(password: str) -> Settings:
    return Settings(
        db_host="localhost",
        db_port=5432,
        db_user="app",
        db_name="testdb",
        db_password=password,
        mqtt_host="localhost",
        mqtt_port=1883,
        mqtt_username="test",
        mqtt_password="test",
        session_signing_key="test-session-key-0123456789abcdef",
        clip_link_signing_key="test-clip-link-key-0123456789abcdef",
    )


def test_database_url_keeps_special_characters_in_password() -> None:
    url = make_settings(password="p@ss/word").database_url

    assert url.password == "p@ss/word"
    assert url.host == "localhost"
    assert url.database == "testdb"


def test_database_url_hides_password_when_printed() -> None:
    url = make_settings(password="supersecret").database_url

    assert "supersecret" not in str(url)
    assert "supersecret" not in repr(url)


def test_migration_url_uses_the_schema_owner() -> None:
    settings = make_settings(password="app-password").model_copy(
        update={
            "db_migration_user": "owner",
            "db_migration_password": SecretStr("owner-password"),
        }
    )

    url = settings.migration_database_url
    assert url.username == "owner"
    assert url.password == "owner-password"
    # The app itself still connects as the limited user.
    assert settings.database_url.username == "app"


def test_migration_url_needs_migration_credentials() -> None:
    # The running app does not get them; only Alembic asks for them.
    settings = make_settings(password="app-password")
    with pytest.raises(RuntimeError, match="DB_MIGRATION_USER"):
        _ = settings.migration_database_url
