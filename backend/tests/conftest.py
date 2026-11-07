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
