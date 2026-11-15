import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import CheckConstraint, LargeBinary, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base, IdMixin

# Both tables are written once and never changed, so they have no
# updated_at or version (no EntityMixin).


class ProcessedEvent(Base):
    """An event BE has already processed; makes redelivery safe (SRS §6.4)."""

    __tablename__ = "processed_events"
    __table_args__ = (
        # sha256 as hex.
        CheckConstraint("char_length(payload_hash) = 64", name="payload_hash_sha256"),
    )

    # ML's own event id, not generated here: a second row with the same id is
    # refused by the primary key itself.
    event_id: Mapped[uuid.UUID] = mapped_column(primary_key=True)
    # Same event_id with another hash is EVENT_ID_CONFLICT (V16).
    payload_hash: Mapped[str]
    # Returned as is on redelivery; its shape is defined by ingest.
    result: Mapped[dict[str, Any]] = mapped_column(JSONB)
    # now() is right here: the row is written in the same transaction as the
    # event's effects (I10). Indexed for retention cleanup (OQ-17).
    processed_at: Mapped[datetime] = mapped_column(
        server_default=func.now(), index=True
    )


class DeadLetter(IdMixin, Base):
    """A rejected event, kept for diagnostics (SRS §7.5)."""

    __tablename__ = "dead_letters"

    # Set by ingest when the message arrives, like Incident.created_received_at.
    received_at: Mapped[datetime] = mapped_column(index=True)
    topic: Mapped[str]
    failure_code: Mapped[str]
    payload_hash: Mapped[str]
    # Raw bytes, not JSONB: a MALFORMED_JSON payload (V1) is not valid JSON,
    # or even valid UTF-8. Ingest removes embeddings before storing (§12.6).
    raw_payload: Mapped[bytes] = mapped_column(LargeBinary)
