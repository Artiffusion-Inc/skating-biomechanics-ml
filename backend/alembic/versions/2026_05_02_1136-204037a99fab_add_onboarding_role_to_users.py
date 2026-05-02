"""add onboarding_role to users

Revision ID: 204037a99fab
Revises: 319070b4dea9
Create Date: 2026-05-02 11:36:54.721846

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "204037a99fab"
down_revision: str | Sequence[str] | None = "319070b4dea9"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column("users", sa.Column("onboarding_role", sa.String(20), nullable=True))


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column("users", "onboarding_role")
