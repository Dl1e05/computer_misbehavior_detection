import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import CheckConstraint, ForeignKey, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.mutable import MutableDict, MutableList
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base, EntityMixin, TimestampMixin
from app.domain.camera import CameraSourceType, CameraStatus


class Camera(EntityMixin, Base):
    __tablename__ = "cameras"

    # Business ID used in ML events as camera_id, e.g. "cam-02".
    code: Mapped[str] = mapped_column(unique=True)
    name: Mapped[str]
    source_type: Mapped[CameraSourceType]
    # Address or file path without credentials; those live in camera_secrets.
    source_ref: Mapped[str]
    status: Mapped[CameraStatus] = mapped_column(
        server_default=CameraStatus.OFFLINE.value
    )
    last_seen_at: Mapped[datetime | None]
    # Administrator switch (BE-FR-60); the only truth for "process this camera".
    enabled: Mapped[bool] = mapped_column(server_default=text("true"))
    # Mutable* tracks in-place edits (settings["k"] = v) so they are saved.
    access_settings: Mapped[dict[str, Any]] = mapped_column(
        MutableDict.as_mutable(JSONB), server_default=text("'{}'::jsonb")
    )


class CameraSecret(TimestampMixin, Base):
    """Reference to a camera's credentials; never returned by the API."""

    __tablename__ = "camera_secrets"

    camera_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("cameras.id", ondelete="CASCADE"), primary_key=True
    )
    # Key into an external secret store or environment; not the password itself.
    secret_ref: Mapped[str]


class Zone(EntityMixin, Base):
    __tablename__ = "zones"
    __table_args__ = (
        CheckConstraint("frame_width > 0", name="frame_width_positive"),
        CheckConstraint("frame_height > 0", name="frame_height_positive"),
        CheckConstraint(
            "CASE WHEN jsonb_typeof(polygon) = 'array' "
            "THEN jsonb_array_length(polygon) >= 3 ELSE false END",
            name="polygon_min_3_points",
        ),
    )

    # Cameras are disabled, not deleted (BE-FR-60), so deleting a camera that
    # still has zones is refused (default ON DELETE behaviour).
    camera_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("cameras.id"), index=True)
    name: Mapped[str]
    # List of [x, y] points in pixels of frame_width x frame_height.
    # MutableList tracks append/replace of points; editing a point in place
    # (polygon[0][0] = 5) is not tracked, so assign a new point instead.
    polygon: Mapped[list[list[int]]] = mapped_column(MutableList.as_mutable(JSONB))
    frame_width: Mapped[int]
    frame_height: Mapped[int]
    active: Mapped[bool] = mapped_column(server_default=text("true"))
