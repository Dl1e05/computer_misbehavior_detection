"""add incidents.episode_id

Links a fight suspicion to its confirmation. The partial unique index allows
one open fight per camera and episode; ended incidents are not counted.

Revision ID: 50397a809a4f
Revises: 265bdce4c331
Create Date: 2026-11-27 18:44:42.913233

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "50397a809a4f"
down_revision: str | Sequence[str] | None = "265bdce4c331"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column("incidents", sa.Column("episode_id", sa.String(), nullable=True))
    # Written out: a migration must not change when TERMINAL_STATUSES does.
    op.create_index(
        "uq_incidents_open_episode",
        "incidents",
        ["camera_id", "episode_id"],
        unique=True,
        postgresql_where=sa.text(
            "status NOT IN ('CLOSED', 'FALSE_ALARM', 'NOT_CONFIRMED')"
        ),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index("uq_incidents_open_episode", table_name="incidents")
    op.drop_column("incidents", "episode_id")
