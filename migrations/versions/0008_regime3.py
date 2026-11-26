"""regime3: the admin-only generation mode (2026-11-26)

Revision ID: 0008
Revises: 0007
"""

from alembic import op

revision = "0008"
down_revision = "0007"
branch_labels = None
depends_on = None

TABLES = ("jobs", "recipes")


def _constraint(table: str, values: tuple[str, ...]) -> None:
    listed = ", ".join(f"'{value}'" for value in values)
    op.execute(f"ALTER TABLE {table} DROP CONSTRAINT IF EXISTS ck_{table}_regime")
    op.execute(f"ALTER TABLE {table} ADD CONSTRAINT ck_{table}_regime CHECK (regime IN ({listed}))")


def upgrade() -> None:
    for table in TABLES:
        _constraint(table, ("multishot", "keyframes", "regime3"))


def downgrade() -> None:
    op.execute("UPDATE jobs SET regime = 'keyframes' WHERE regime = 'regime3'")
    for table in TABLES:
        _constraint(table, ("multishot", "keyframes"))
