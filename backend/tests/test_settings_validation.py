import os

import pytest
from pydantic import ValidationError

from app.config import Settings

LONG_KEY_A = "a" * 32
LONG_KEY_B = "b" * 32


def make_settings(session_key: str, clip_key: str) -> Settings:
    return Settings(
        db_host="localhost",
        db_port=5432,
        db_user="app",
        db_name="testdb",
        db_password="test",
        mqtt_host="localhost",
        mqtt_port=1883,
        mqtt_username="test",
        mqtt_password="test",
        session_signing_key=session_key,
        clip_link_signing_key=clip_key,
    )


def test_valid_signing_keys_are_accepted() -> None:
    settings = make_settings(LONG_KEY_A, LONG_KEY_B)
    assert settings.session_signing_key.get_secret_value() == LONG_KEY_A


def test_short_signing_key_is_rejected_without_echoing_it() -> None:
    with pytest.raises(ValidationError) as error:
        make_settings("short-secret", LONG_KEY_B)

    assert "session_signing_key" in str(error.value)
    assert "short-secret" not in str(error.value)


def test_signing_keys_must_differ() -> None:
    with pytest.raises(ValidationError, match="must differ"):
        make_settings(LONG_KEY_A, LONG_KEY_A)


def test_shell_leftovers_are_removed_from_environment() -> None:
    # The autouse fixture deletes every Settings variable that tests do not
    # set explicitly, so values exported in the developer's shell (e.g. after
    # `source .env`) cannot change test results.
    leftovers = [
        name.upper()
        for name, field in Settings.model_fields.items()
        if not field.is_required() and name.upper() in os.environ
    ]
    assert leftovers == []
