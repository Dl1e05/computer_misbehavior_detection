import uuid
from datetime import datetime

from sqlalchemy import (
    CheckConstraint,
    ForeignKey,
    Index,
    Integer,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base, EntityMixin
from app.domain.incident import (
    SEVERITY_BY_TYPE,
    ClipState,
    IncidentStatus,
    IncidentType,
    Severity,
    TimerKind,
    TimerState,
)

# ('zone_intrusion', 'MEDIUM'), ('fall', 'HIGH'), ... built from the domain map,
# so the database rule cannot drift from severity_for().
_SEVERITY_PAIRS = ", ".join(
    f"('{incident_type.value}', '{severity.value}')"
    for incident_type, severity in SEVERITY_BY_TYPE.items()
)

# Built from the enum, so the partial indexes cannot silently match nothing.
_TIMER_PENDING = f"state = '{TimerState.PENDING.value}'"


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


class Participant(EntityMixin, Base):
    """A person ML tracked in an incident; repeat events update the same row."""

    __tablename__ = "participants"
    __table_args__ = (
        # One row per tracked person per incident. Also serves as the index
        # for incident_id, since it is the first column.
        UniqueConstraint("incident_id", "track_id"),
        CheckConstraint(
            "confidence >= 0 AND confidence <= 1", name="confidence_in_range"
        ),
        CheckConstraint("cardinality(bbox) = 4", name="bbox_has_4_values"),
    )

    # A participant means nothing without its incident.
    incident_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("incidents.id", ondelete="CASCADE")
    )
    # Tracker id from ML, unique only within one incident.
    track_id: Mapped[str]
    confidence: Mapped[float]
    # [x, y, width, height] in frame pixels, as sent by ML.
    bbox: Mapped[list[int]] = mapped_column(ARRAY(Integer))
    # Link to the stored snapshot, not the image itself.
    snapshot_ref: Mapped[str | None]
    model_version: Mapped[str]


class Timer(EntityMixin, Base):
    """A deadline for an incident; the scheduler fires it (SRS §6.5)."""

    __tablename__ = "timers"
    __table_args__ = (
        # Old FIRED/CANCELLED timers stay as history, so only PENDING rows
        # are unique per incident and kind.
        Index(
            "uq_timers_one_pending_per_kind",
            "incident_id",
            "kind",
            unique=True,
            postgresql_where=text(_TIMER_PENDING),
        ),
        # The scheduler polls "PENDING and deadline_at <= now()" every second.
        Index(
            "ix_timers_pending_deadline",
            "deadline_at",
            postgresql_where=text(_TIMER_PENDING),
        ),
        CheckConstraint(
            f"(state = '{TimerState.FIRED.value}') = (fired_at IS NOT NULL)",
            name="fired_at_iff_fired",
        ),
    )

    incident_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("incidents.id", ondelete="CASCADE")
    )

    kind: Mapped[TimerKind]
    state: Mapped[TimerState] = mapped_column(server_default=TimerState.PENDING.value)

    deadline_at: Mapped[datetime]
    fired_at: Mapped[datetime | None]


class Escalation(EntityMixin, Base):
    """An escalation level raised for an incident (SRS §6.4)."""

    __tablename__ = "escalations"
    __table_args__ = (
        UniqueConstraint("incident_id", "level"),
        CheckConstraint("level > 0", name="level_positive"),
        CheckConstraint(
            "(acknowledged_by IS NULL) = (acknowledged_at IS NULL)",
            name="ack_fields_together",
        ),
    )

    incident_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("incidents.id", ondelete="CASCADE")
    )
    level: Mapped[int]
    raised_at: Mapped[datetime] = mapped_column(server_default=func.now())

    acknowledged_by: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("users.id"))
    acknowledged_at: Mapped[datetime | None]
