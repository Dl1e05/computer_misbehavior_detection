"""The ML → BE event contract, as received over MQTT (SRS §7.2, §7.3).

These models check what a message can be checked against on its own.
Checks that need the database (camera, zone ownership, idempotency,
transition) belong to event processing.
"""

import uuid

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field

from app.domain.event import EventType
from app.domain.incident import IncidentType

# extra="ignore": V3 accepts any minor version of a supported major one, so
# a field added in schema 1.1 must not make BE reject the whole event.
# strict: "0.9" is not a number and 1 is not true; JSON strings still parse
# into UUID, datetime and enum fields.
_CONTRACT_CONFIG = ConfigDict(extra="ignore", strict=True, frozen=True)


class Participant(BaseModel):
    model_config = _CONTRACT_CONFIG

    # ByteTrack id, stable within the camera session.
    track_id: str = Field(min_length=1)
    # Detection or track confidence; not an identity score.
    confidence: float = Field(ge=0, le=1)
    # [x1, y1, x2, y2]. Units (pixels or 0..1) are pending with the ML team.
    bbox: list[float] | None = Field(default=None, min_length=4, max_length=4)
    # Storage key of the face snapshot, never a URL.
    snapshot_ref: str | None = None
    # Face embedding; allowed only on some events (V14), D long (V15).
    embedding: list[float] | None = None
    embedding_model_version: str | None = None


class MlEvent(BaseModel):
    model_config = _CONTRACT_CONFIG

    schema_version: str = Field(min_length=1)
    # Reused on redelivery of the same event (idempotency key).
    event_id: uuid.UUID
    event_type: EventType
    camera_id: str = Field(min_length=1)
    # AwareDatetime refuses a time without a timezone.
    occurred_at: AwareDatetime
    incident_type: IncidentType
    zone_id: str | None = None
    # Detection pipeline version, e.g. "fight-temporal-1.3.0/yolo26-det-2.0.1".
    model_version: str = Field(min_length=1, max_length=128)
    participants: list[Participant] = Field(default_factory=list)
    sent_at: AwareDatetime | None = None
    # Links a fight suspicion to its confirmation (proposed by the ML team).
    episode_id: str | None = None
