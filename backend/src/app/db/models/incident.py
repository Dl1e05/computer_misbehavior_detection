import uuid
from datetime import datetime

from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    Float,
    ForeignKey,
    Index,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base, EntityMixin
from app.domain.incident import (
    SEVERITY_BY_TYPE,
    ClipRecordState,
    ClipState,
    IncidentStatus,
    IncidentType,
    Severity,
    TimerKind,
    TimerState,
)

# ('zone_intrusion', 'MEDIUM'), ('fall', 'HIGH'), ... built from the domain map.
# Autogenerate does not compare CHECK bodies, so changing SEVERITY_BY_TYPE
# needs a hand-written migration; test_severity_check_matches_domain_map
# fails until the migrated database matches.
_SEVERITY_PAIRS = ", ".join(
    f"('{incident_type.value}', '{severity.value}')"
    for incident_type, severity in SEVERITY_BY_TYPE.items()
)

# [x1, y1, x2, y2] as fractions of the frame; arrays are 1-indexed. array_position
# is needed: "0 <= ALL" with a NULL element is NULL, and a NULL CHECK passes.
_BBOX_NORMALIZED = (
    "bbox IS NULL OR (array_position(bbox, NULL) IS NULL "
    "AND 0 <= ALL (bbox) AND 1 >= ALL (bbox) "
    "AND bbox[1] < bbox[3] AND bbox[2] < bbox[4])"
)

# Built from the enum, so the partial indexes cannot silently match nothing.
_TIMER_PENDING = f"state = '{TimerState.PENDING.value}'"


class Incident(EntityMixin, Base):
    __tablename__ = "incidents"
    __table_args__ = (
        # I7 (SRS §12.5): an incident cannot be closed without measures taken.
        CheckConstraint(
            # The regex means "has a non-whitespace character" (btrim only
            # strips spaces). IS NOT NULL is needed: a NULL CHECK result passes.
            f"status <> '{IncidentStatus.CLOSED.value}' "
            "OR (measures IS NOT NULL AND measures ~ '\\S')",
            name="closed_requires_measures",
        ),
        # Severity is fixed by type (SRS §5); a service bug cannot store another.
        CheckConstraint(
            f"(incident_type, severity) IN ({_SEVERITY_PAIRS})",
            name="severity_matches_type",
        ),
        # V11 (SRS §7.4). That the zone belongs to the camera is checked
        # at ingest.
        CheckConstraint(
            f"incident_type <> '{IncidentType.ZONE_INTRUSION.value}' "
            "OR zone_id IS NOT NULL",
            name="zone_intrusion_requires_zone",
        ),
        CheckConstraint("cardinality(bbox) = 4", name="bbox_has_4_values"),
        CheckConstraint(_BBOX_NORMALIZED, name="bbox_normalized"),
        CheckConstraint(
            "confidence >= 0 AND confidence <= 1", name="confidence_in_range"
        ),
    )

    incident_type: Mapped[IncidentType]
    severity: Mapped[Severity]
    status: Mapped[IncidentStatus] = mapped_column(index=True)
    camera_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("cameras.id"), index=True)
    zone_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("zones.id"), index=True
    )
    # Version of the zone ML used; the zone may have been edited since.
    zone_version: Mapped[int | None]
    # Detection pipeline version. On the incident too, because a fire has no
    # participants.
    model_version: Mapped[str]
    # Confidence of the whole event; not an identity score.
    confidence: Mapped[float | None]
    # Where the fire is; other types have boxes on their participants.
    bbox: Mapped[list[float] | None] = mapped_column(ARRAY(Float))
    measures: Mapped[str | None]
    clip_state: Mapped[ClipState] = mapped_column(
        server_default=ClipState.PENDING.value
    )

    # Time from the ML event; no default, so a missing value fails loudly.
    occurred_at: Mapped[datetime]
    # Time BE received the event; timers count from this one (SRS §6.1).
    # Set by ingest when the message arrives: now() would be the transaction
    # start, which is later and would only duplicate created_at.
    created_received_at: Mapped[datetime]

    # status timestamps
    in_progress_at: Mapped[datetime | None]
    confirmed_at: Mapped[datetime | None]
    rejected_at: Mapped[datetime | None]
    closed_at: Mapped[datetime | None]


class Participant(EntityMixin, Base):
    """A person ML tracked in an incident; repeat events update the same row."""

    __tablename__ = "participants"
    __table_args__ = (
        # One row per tracked person per incident; track ids start again in
        # each run of ML. Also serves as the index for incident_id, since it
        # is the first column.
        UniqueConstraint("incident_id", "run_id", "track_id"),
        CheckConstraint(
            "confidence >= 0 AND confidence <= 1", name="confidence_in_range"
        ),
        CheckConstraint("cardinality(bbox) = 4", name="bbox_has_4_values"),
        CheckConstraint(_BBOX_NORMALIZED, name="bbox_normalized"),
    )

    # A participant means nothing without its incident.
    incident_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("incidents.id", ondelete="CASCADE")
    )
    # The run of the ML process that gave the track id.
    run_id: Mapped[str]
    # Tracker id from ML (a number, kept as text), unique only within one run.
    track_id: Mapped[str]
    confidence: Mapped[float]
    # [x1, y1, x2, y2] as fractions of the frame (0..1); optional (SRS §7.3).
    bbox: Mapped[list[float] | None] = mapped_column(ARRAY(Float))
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

    # Full index: the partial ones above skip FIRED/CANCELLED history rows.
    incident_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("incidents.id", ondelete="CASCADE"), index=True
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


class Clip(EntityMixin, Base):
    """The archived video fragment of a confirmed incident (SRS §13)."""

    __tablename__ = "clips"
    __table_args__ = (
        CheckConstraint("end_at > start_at", name="end_after_start"),
        # An available clip must be findable and verifiable. coalesce makes
        # NULL fail too (a bare "location <> ''" is NULL, and NULL passes).
        CheckConstraint(
            f"state <> '{ClipRecordState.AVAILABLE.value}' "
            "OR (coalesce(location, '') <> '' AND coalesce(checksum, '') <> '' "
            "AND coalesce(size_bytes, 0) > 0)",
            name="available_has_file",
        ),
        CheckConstraint(
            f"(state = '{ClipRecordState.ERROR.value}') = (error_code IS NOT NULL)",
            name="error_code_iff_error",
        ),
        CheckConstraint("size_bytes >= 0", name="size_not_negative"),
    )

    # One clip per incident (BE-FR-48). No CASCADE: the row points to a file
    # in the archive, so the deletion procedure removes the file first.
    incident_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("incidents.id"), unique=True
    )
    # Storage key in the clip archive, never a URL.
    location: Mapped[str | None]
    # Trigger minus pre-roll / plus post-roll (SRS §13.4).
    start_at: Mapped[datetime]
    end_at: Mapped[datetime]
    state: Mapped[ClipRecordState] = mapped_column(
        server_default=ClipRecordState.PENDING.value
    )
    error_code: Mapped[str | None]
    # BigInteger: a plain integer column stops at about 2 GB.
    size_bytes: Mapped[int | None] = mapped_column(BigInteger)
    checksum: Mapped[str | None]
