"""published versions cannot change

Revision ID: 0002
Revises: 0001
"""

from alembic import op

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None

# The columns that name one thing; at most one of its versions is active.
KEYS = {
    "workflow_versions": "name",
    "prompt_versions": "role, model",
    "engines": "slot, name",
    "recipes": "name",
}

# Every column except status is frozen, so a column added later is frozen too. The comparison
# as jsonb makes 1 and 1.0 equal, as the sha256 of a workflow does.
GUARD = """
CREATE OR REPLACE FUNCTION guard_version_row() RETURNS trigger LANGUAGE plpgsql AS $guard$
BEGIN
    IF TG_OP = 'DELETE' THEN
        IF OLD.status <> 'draft' THEN
            RAISE EXCEPTION USING ERRCODE = 'check_violation',
                MESSAGE = 'a published row of ' || TG_TABLE_NAME || ' cannot be deleted';
        END IF;
        RETURN OLD;
    END IF;
    IF OLD.status <> 'draft'
            AND to_jsonb(NEW) - 'status' IS DISTINCT FROM to_jsonb(OLD) - 'status' THEN
        RAISE EXCEPTION USING ERRCODE = 'check_violation',
            MESSAGE = 'a published row of ' || TG_TABLE_NAME || ' cannot be edited';
    END IF;
    IF NEW.status <> OLD.status AND (OLD.status, NEW.status) NOT IN (
            ('draft', 'candidate'), ('candidate', 'active'),
            ('candidate', 'retired'), ('active', 'retired')) THEN
        RAISE EXCEPTION USING ERRCODE = 'check_violation',
            MESSAGE = 'a row of ' || TG_TABLE_NAME || ' cannot go from ' || OLD.status
                || ' to ' || NEW.status;
    END IF;
    RETURN NEW;
END
$guard$
"""


def upgrade() -> None:
    op.execute(GUARD)
    for table, key in KEYS.items():
        op.execute(
            f"CREATE TRIGGER {table}_guard BEFORE UPDATE OR DELETE ON {table} "
            "FOR EACH ROW EXECUTE FUNCTION guard_version_row()"
        )
        op.execute(
            f"CREATE UNIQUE INDEX uq_{table}_active ON {table} ({key}) WHERE status = 'active'"
        )


def downgrade() -> None:
    for table in reversed(KEYS):
        op.execute(f"DROP INDEX uq_{table}_active")
        op.execute(f"DROP TRIGGER {table}_guard ON {table}")
    op.execute("DROP FUNCTION guard_version_row()")
