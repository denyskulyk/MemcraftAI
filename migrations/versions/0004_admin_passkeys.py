"""admin passkeys, sign-in challenges and admin sessions

Revision ID: 0004
Revises: 0003
"""

import sqlalchemy as sa
from alembic import op

revision = "0004"
down_revision = "0003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "passkeys",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("user_id", sa.Uuid(), nullable=False),
        sa.Column("credential_id", sa.LargeBinary(), nullable=False),
        sa.Column("public_key", sa.LargeBinary(), nullable=False),
        sa.Column("sign_count", sa.BigInteger(), server_default=sa.text("0"), nullable=False),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("last_used_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(
            ["user_id"], ["users.id"], name=op.f("fk_passkeys_user_id"), ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_passkeys")),
        sa.UniqueConstraint("credential_id", name=op.f("uq_passkeys_credential_id")),
    )
    op.create_index(op.f("ix_passkeys_user_id"), "passkeys", ["user_id"], unique=False)
    op.create_table(
        "auth_challenges",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("user_id", sa.Uuid(), nullable=False),
        sa.Column(
            "purpose",
            sa.Enum(
                "register",
                "sign_in",
                name="purpose",
                native_enum=False,
                create_constraint=True,
                length=32,
            ),
            nullable=False,
        ),
        sa.Column("challenge", sa.LargeBinary(), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["user_id"], ["users.id"], name=op.f("fk_auth_challenges_user_id"), ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_auth_challenges")),
    )
    op.create_index(
        op.f("ix_auth_challenges_user_id"), "auth_challenges", ["user_id"], unique=False
    )
    op.create_table(
        "admin_sessions",
        sa.Column("token_sha256", sa.String(), nullable=False),
        sa.Column("user_id", sa.Uuid(), nullable=False),
        sa.Column("passkey_id", sa.Uuid(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["passkey_id"],
            ["passkeys.id"],
            name=op.f("fk_admin_sessions_passkey_id"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["user_id"], ["users.id"], name=op.f("fk_admin_sessions_user_id"), ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("token_sha256", name=op.f("pk_admin_sessions")),
    )
    op.create_index(op.f("ix_admin_sessions_user_id"), "admin_sessions", ["user_id"], unique=False)


def downgrade() -> None:
    op.drop_table("admin_sessions")
    op.drop_table("auth_challenges")
    op.drop_table("passkeys")
