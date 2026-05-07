"""add session imu_keys

Revision ID: d4e5f6a7b8c9
Revises: c7d8e9f0a1b2
Create Date: 2026-05-07 18:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "d4e5f6a7b8c9"
down_revision: str | None = "c7d8e9f0a1b2"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("sessions", sa.Column("imu_left_key", sa.String(500), nullable=True))
    op.add_column("sessions", sa.Column("imu_right_key", sa.String(500), nullable=True))
    op.add_column("sessions", sa.Column("manifest_key", sa.String(500), nullable=True))


def downgrade() -> None:
    op.drop_column("sessions", "manifest_key")
    op.drop_column("sessions", "imu_right_key")
    op.drop_column("sessions", "imu_left_key")
