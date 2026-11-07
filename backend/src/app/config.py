from functools import lru_cache
from pathlib import Path
from typing import Literal, Self

from pydantic import Field, PositiveInt, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=Path(__file__).resolve().parents[2] / ".env",
        env_file_encoding="utf-8",
        env_ignore_empty=True,
        extra="forbid",
        # Never echo rejected values (e.g. a too-short key) into error messages.
        hide_input_in_errors=True,
    )

    # Timers, in seconds
    timer_suspicion_seconds: PositiveInt = 10
    timer_escalation_seconds: PositiveInt = 30
    repeat_window_seconds: PositiveInt = 60
    clip_pre_roll_seconds: int = Field(default=5, ge=0)
    clip_post_roll_seconds: int = Field(default=5, ge=0)
    clock_skew_warn_seconds: int = Field(default=2, ge=0)
    signed_link_ttl_seconds: PositiveInt = 60

    # Identification
    ident_top_k: PositiveInt = 5
    ident_match_threshold: float | None = Field(default=None, ge=0, le=1)
    ident_margin: float | None = Field(default=None, ge=0, le=1)

    # Event processing
    event_ack_mode: Literal["after_commit"] = "after_commit"

    # Database connection
    db_host: str
    db_port: PositiveInt
    db_user: str
    db_name: str
    db_password: SecretStr

    # MQTT connection
    mqtt_host: str
    mqtt_port: PositiveInt
    mqtt_username: str
    mqtt_password: SecretStr

    # Independent signing secrets (SRS §14.4): long enough for HMAC-SHA256
    # and never the same, so a clip link can't pass as a session token.
    session_signing_key: SecretStr = Field(min_length=32)
    clip_link_signing_key: SecretStr = Field(min_length=32)

    @model_validator(mode="after")
    def _signing_keys_differ(self) -> Self:
        if (
            self.session_signing_key.get_secret_value()
            == self.clip_link_signing_key.get_secret_value()
        ):
            raise ValueError(
                "SESSION_SIGNING_KEY and CLIP_LINK_SIGNING_KEY must differ"
            )
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
