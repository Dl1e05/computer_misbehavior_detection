"""switch to normalized coordinates and run_id

Zones and boxes are fractions of the frame (0..1) instead of pixels; track ids
are numbers and belong to a run of ML. Existing rows are not converted: nothing creates
them yet, and the migration fails (changing nothing) if the tables have any.

Revision ID: 265bdce4c331
Revises: dc79901129d9
Create Date: 2026-11-24 16:48:10.513203

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "265bdce4c331"
down_revision: str | Sequence[str] | None = "dc79901129d9"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


POLYGON_NORMALIZED = (
    "coalesce(NOT jsonb_path_exists(polygon, 'strict $[*] ? "
    '(@.type() != "array" || @.size() != 2 || '
    'exists (@[*] ? (@.type() != "number" || @ < 0 || @ > 1)))'
    "', '{}', true), false)"
)
BBOX_NORMALIZED = (
    "bbox IS NULL OR (array_position(bbox, NULL) IS NULL "
    "AND 0 <= ALL (bbox) AND 1 >= ALL (bbox) "
    "AND bbox[1] <= bbox[3] AND bbox[2] <= bbox[4])"
)


def upgrade() -> None:
    """Upgrade schema."""
    # Autogenerate does not see CHECKs; all of them are written by hand.
    op.create_check_constraint(
        op.f("ck_zones_polygon_normalized"), "zones", POLYGON_NORMALIZED
    )
    op.drop_constraint(op.f("ck_zones_frame_height_positive"), "zones", type_="check")
    op.drop_constraint(op.f("ck_zones_frame_width_positive"), "zones", type_="check")
    op.drop_column("zones", "frame_height")
    op.drop_column("zones", "frame_width")

    op.add_column("incidents", sa.Column("zone_version", sa.Integer(), nullable=True))
    op.add_column("incidents", sa.Column("model_version", sa.String(), nullable=False))
    op.add_column("incidents", sa.Column("confidence", sa.Double(), nullable=True))
    op.add_column(
        "incidents", sa.Column("bbox", postgresql.ARRAY(sa.Float()), nullable=True)
    )
    op.create_check_constraint(
        op.f("ck_incidents_bbox_has_4_values"), "incidents", "cardinality(bbox) = 4"
    )
    op.create_check_constraint(
        op.f("ck_incidents_bbox_normalized"), "incidents", BBOX_NORMALIZED
    )
    op.create_check_constraint(
        op.f("ck_incidents_confidence_in_range"),
        "incidents",
        "confidence >= 0 AND confidence <= 1",
    )

    op.add_column("participants", sa.Column("run_id", sa.String(), nullable=False))
    op.drop_constraint(
        op.f("uq_participants_incident_id_track_id"), "participants", type_="unique"
    )
    op.create_unique_constraint(
        op.f("uq_participants_incident_id_run_id_track_id"),
        "participants",
        ["incident_id", "run_id", "track_id"],
    )
    op.create_check_constraint(
        op.f("ck_participants_bbox_normalized"), "participants", BBOX_NORMALIZED
    )
    op.alter_column(
        "participants",
        "track_id",
        type_=sa.BigInteger(),
        existing_type=sa.String(),
        existing_nullable=False,
        postgresql_using="track_id::bigint",
    )
    op.create_check_constraint(
        op.f("ck_participants_track_id_not_negative"), "participants", "track_id >= 0"
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_constraint(
        op.f("ck_participants_track_id_not_negative"), "participants", type_="check"
    )
    op.alter_column(
        "participants",
        "track_id",
        type_=sa.String(),
        existing_type=sa.BigInteger(),
        existing_nullable=False,
        postgresql_using="track_id::varchar",
    )
    op.drop_constraint(
        op.f("ck_participants_bbox_normalized"), "participants", type_="check"
    )
    op.drop_constraint(
        op.f("uq_participants_incident_id_run_id_track_id"),
        "participants",
        type_="unique",
    )
    op.create_unique_constraint(
        op.f("uq_participants_incident_id_track_id"),
        "participants",
        ["incident_id", "track_id"],
    )
    op.drop_column("participants", "run_id")

    # Dropping a column also drops the CHECKs that use only that column.
    op.drop_column("incidents", "bbox")
    op.drop_column("incidents", "confidence")
    op.drop_column("incidents", "model_version")
    op.drop_column("incidents", "zone_version")

    op.drop_constraint(op.f("ck_zones_polygon_normalized"), "zones", type_="check")
    op.add_column("zones", sa.Column("frame_width", sa.Integer(), nullable=False))
    op.add_column("zones", sa.Column("frame_height", sa.Integer(), nullable=False))
    op.create_check_constraint(
        op.f("ck_zones_frame_width_positive"), "zones", "frame_width > 0"
    )
    op.create_check_constraint(
        op.f("ck_zones_frame_height_positive"), "zones", "frame_height > 0"
    )
