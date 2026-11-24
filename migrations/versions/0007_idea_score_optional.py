"""ideas.score is optional: ideas are no longer scored (2026-11-24)

Revision ID: 0007
Revises: 0006
"""

import sqlalchemy as sa
from alembic import op

revision = "0007"
down_revision = "0006"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.alter_column("ideas", "score", existing_type=sa.Integer(), nullable=True)


def downgrade() -> None:
    op.execute("UPDATE ideas SET score = 5 WHERE score IS NULL")
    op.alter_column("ideas", "score", existing_type=sa.Integer(), nullable=False)
