"""the order of a job's photos, keyframes and clips

Revision ID: 0003
Revises: 0002
"""

import sqlalchemy as sa
from alembic import op

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "assets", sa.Column("ordinal", sa.Integer(), server_default=sa.text("0"), nullable=False)
    )


def downgrade() -> None:
    op.drop_column("assets", "ordinal")
