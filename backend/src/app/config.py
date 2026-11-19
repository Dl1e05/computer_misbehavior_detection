from functools import lru_cache
from pathlib import Path
from typing import Literal, Self

from pydantic import Field, PositiveInt, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy import URL


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
    # V9: an event further in the future than this is rejected; smaller skew
    # is only a warning (V10). Proposed value, the SRS sets none.
    event_max_future_seconds: PositiveInt = 60
    signed_link_ttl_seconds: PositiveInt = 60

    # Identification
    ident_top_k: PositiveInt = 5
    ident_match_threshold: float | None = Field(default=None, ge=0, le=1)
    ident_margin: float | None = Field(default=None, ge=0, le=1)
    # Length of a face embedding (InsightFace: 512 assumed, to be confirmed
    # by the ML team).
    embedding_dim: PositiveInt = 512

    # Event processing
    event_ack_mode: Literal["after_commit"] = "after_commit"

    # Database connection. DB_USER is the limited app login (member of the
    # app_runtime role); it must not own the schema (BE-FR-75).
    db_host: str
    db_port: int = Field(ge=1, le=65535)
    db_user: str
    db_name: str
    db_password: SecretStr
    # Schema owner, used only by Alembic (migrations/env.py). Optional: the
    # running app does not need, and should not get, the owner password.
    db_migration_user: str | None = None
    db_migration_password: SecretStr | None = None

    # MQTT connection
    mqtt_host: str
    mqtt_port: int = Field(ge=1, le=65535)
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

    @model_validator(mode="after")
    def _app_is_not_schema_owner(self) -> Self:
        # An owner ignores GRANT/REVOKE, so the append-only audit would
        # silently stop being append-only.
        if (
            self.db_migration_user is not None
            and self.db_user == self.db_migration_user
        ):
            raise ValueError(
                "DB_USER must differ from DB_MIGRATION_USER: the app must not "
                "connect as the schema owner"
            )
        return self

    @model_validator(mode="after")
    def _future_limit_above_warning(self) -> Self:
        # Skew up to the warning threshold is only a warning (V10), so the
        # rejection threshold (V9) must not be below it.
        if self.event_max_future_seconds < self.clock_skew_warn_seconds:
            raise ValueError(
                "EVENT_MAX_FUTURE_SECONDS must be >= CLOCK_SKEW_WARN_SECONDS"
            )
        return self

    @property
    def database_url(self) -> URL:
        return self._url(self.db_user, self.db_password)

    @property
    def migration_database_url(self) -> URL:
        if self.db_migration_user is None or self.db_migration_password is None:
            raise RuntimeError(
                "DB_MIGRATION_USER and DB_MIGRATION_PASSWORD are required "
                "to run migrations"
            )
        return self._url(self.db_migration_user, self.db_migration_password)

    def _url(self, user: str, password: SecretStr) -> URL:
        return URL.create(
            drivername="postgresql+asyncpg",
            username=user,
            password=password.get_secret_value(),
            host=self.db_host,
            port=self.db_port,
            database=self.db_name,
        )


@lru_cache
def get_settings() -> Settings:
    return Settings()
