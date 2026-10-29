from collections.abc import Iterator

import pytest

from app.config import Settings, get_settings

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
    "SESSION_SIGNING_KEY": "test-session-key",
    "CLIP_LINK_SIGNING_KEY": "test-clip-link-key",
}


@pytest.fixture(autouse=True)
def _test_settings(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Provide deterministic settings without loading a local .env file."""
    monkeypatch.setitem(Settings.model_config, "env_file", None)

    for name, value in REQUIRED_TEST_ENV.items():
        monkeypatch.setenv(name, value)

    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
