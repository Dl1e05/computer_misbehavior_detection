import uuid
from datetime import datetime

from sqlalchemy import CheckConstraint, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base, EntityMixin
from app.domain.incident import Role
from app.domain.notification import INCIDENT_NOTIFICATION_KINDS, NotificationKind

# 'alarm', 'escalation', ... sorted, so the SQL text is the same on every run
# (set order is not stable).
_INCIDENT_KINDS = ", ".join(
    f"'{kind.value}'" for kind in sorted(INCIDENT_NOTIFICATION_KINDS)
)


class Notification(EntityMixin, Base):
    """A realtime message sent to a role; a log for delivery checks (SRS §10)."""

    __tablename__ = "notifications"
    __table_args__ = (
        CheckConstraint(
            f"(kind IN ({_INCIDENT_KINDS})) = (incident_id IS NOT NULL)",
            name="incident_id_iff_incident_kind",
        ),
    )

    # Null for camera and analytics status messages.
    incident_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("incidents.id", ondelete="CASCADE"), index=True
    )
    kind: Mapped[NotificationKind]
    target_role: Mapped[Role]
    # Reference to the message payload, never biometric data (SRS §9.4).
    payload_ref: Mapped[str]
    delivered_at: Mapped[datetime | None]
