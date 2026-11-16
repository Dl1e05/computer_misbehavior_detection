import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import BigInteger, CheckConstraint, Identity, Index, func, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base
from app.domain.audit import AuditObjectType, AuditResult
from app.domain.incident import Role


class AuditEntry(Base):
    """One action in the append-only Audit Log (SRS §12.2, BE-FR-73..75).

    Never updated or deleted, so no EntityMixin. The application database
    role may only INSERT and SELECT here (BE-FR-75).
    """

    __tablename__ = "audit_entries"
    __table_args__ = (
        # NULL actor = system action; then there is no role either.
        CheckConstraint(
            "(actor_id IS NULL) = (actor_role IS NULL)",
            name="actor_fields_together",
        ),
        # "All entries for incident X" (BE-FR-76).
        Index("ix_audit_entries_object", "object_type", "object_id"),
    )

    # Increasing numbers: the order is clear and a gap stands out. ALWAYS:
    # only the database assigns ids; an explicit id from the app is refused.
    id: Mapped[int] = mapped_column(BigInteger, Identity(always=True), primary_key=True)
    # No FK to users: the audit is a historical record and must never block
    # or break because of another table.
    actor_id: Mapped[uuid.UUID | None]
    # Role at the time of the action.
    actor_role: Mapped[Role | None]
    # e.g. incident.transition, clip.access, auth.denied; the list grows.
    action: Mapped[str]
    object_type: Mapped[AuditObjectType]
    # Text: objects have different id types (UUID, camera code, ...).
    object_id: Mapped[str | None]
    result: Mapped[AuditResult]
    # No personal data here.
    reason: Mapped[str | None]
    # Request id, client type, event id where relevant.
    access_context: Mapped[dict[str, Any]] = mapped_column(
        JSONB, server_default=text("'{}'::jsonb")
    )
    # Server time; written in the same transaction as the action (I2).
    occurred_at: Mapped[datetime] = mapped_column(server_default=func.now(), index=True)
