import uuid
from datetime import datetime

from sqlalchemy import CheckConstraint, ForeignKey, func
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base, EntityMixin
from app.domain.incident import (
    SEVERITY_BY_TYPE,
    ClipState,
    IncidentStatus,
    IncidentType,
    Severity,
)

# ('zone_intrusion', 'MEDIUM'), ('fall', 'HIGH'), ... built from the domain map,
# so the database rule cannot drift from severity_for().
_SEVERITY_PAIRS = ", ".join(
    f"('{incident_type.value}', '{severity.value}')"
    for incident_type, severity in SEVERITY_BY_TYPE.items()
)


class Incident(EntityMixin, Base):
    __tablename__ = "incidents"
    __table_args__ = (
        # I7 (SRS §12.5): an incident cannot be closed without measures taken.
        CheckConstraint(
            f"status <> '{IncidentStatus.CLOSED.value}' "
            "OR (measures IS NOT NULL AND btrim(measures) <> '')",
            name="closed_requires_measures",
        ),
        # Severity is fixed by type (SRS §5); a service bug cannot store another.
        CheckConstraint(
            f"(incident_type, severity) IN ({_SEVERITY_PAIRS})",
            name="severity_matches_type",
        ),
    )

    incident_type: Mapped[IncidentType]
    severity: Mapped[Severity]
    status: Mapped[IncidentStatus] = mapped_column(index=True)
    camera_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("cameras.id"), index=True)
    zone_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("zones.id"), index=True
    )
    measures: Mapped[str | None]
    clip_state: Mapped[ClipState] = mapped_column(
        server_default=ClipState.PENDING.value
    )

    # Time from the ML event; no default, so a missing value fails loudly.
    occurred_at: Mapped[datetime]
    # Time BE received the event; timers count from this one (SRS §6.1).
    created_received_at: Mapped[datetime] = mapped_column(server_default=func.now())

    # status timestamps
    in_progress_at: Mapped[datetime | None]
    confirmed_at: Mapped[datetime | None]
    rejected_at: Mapped[datetime | None]
    closed_at: Mapped[datetime | None]
