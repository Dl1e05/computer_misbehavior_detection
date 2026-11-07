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
