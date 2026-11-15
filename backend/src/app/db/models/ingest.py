import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import CheckConstraint, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base, IdMixin

# Both tables are written once and never changed, so they have no
# updated_at or version (no EntityMixin).

# Lowercase hex sha256, the format of hashlib.sha256(raw).hexdigest(). One
# format in both tables, so a dead letter can be matched to a processed event.
_SHA256_HEX = "payload_hash ~ '^[0-9a-f]{64}$'"


class ProcessedEvent(Base):
    """An event BE has already processed; makes redelivery safe (SRS §6.4)."""

    __tablename__ = "processed_events"
    __table_args__ = (
        CheckConstraint(_SHA256_HEX, name="payload_hash_sha256"),
        # NOT NULL does not stop Python None: JSONB stores it as JSON null.
        CheckConstraint("jsonb_typeof(result) = 'object'", name="result_is_object"),
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
    __table_args__ = (CheckConstraint(_SHA256_HEX, name="payload_hash_sha256"),)

    # Set by ingest when the message arrives, like Incident.created_received_at.
    received_at: Mapped[datetime] = mapped_column(index=True)
    topic: Mapped[str]
    failure_code: Mapped[str]
    payload_hash: Mapped[str]
    # No embedding may be stored here (SRS §12.6, §14). Ingest stores the
    # payload only after parsing it and removing embeddings, and within the
    # size limit; otherwise (e.g. MALFORMED_JSON) NULL, and payload_hash is
    # all that is kept. So whatever is stored is valid JSON.
    raw_payload: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
