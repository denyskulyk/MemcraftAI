"""schema v1

Revision ID: 0001
Revises:
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "file_deletions",
        sa.Column("id", sa.BigInteger(), nullable=False),
        sa.Column(
            "store",
            sa.Enum(
                "shared",
                "private",
                name="store",
                native_enum=False,
                create_constraint=True,
                length=32,
            ),
            nullable=False,
        ),
        sa.Column("path", sa.String(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_file_deletions")),
        sa.UniqueConstraint("store", "path", name=op.f("uq_file_deletions_store_path")),
    )
    op.create_table(
        "prompt_versions",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column(
            "role",
            sa.Enum(
                "chat",
                "director_multishot",
                "director_keyframes",
                "critic",
                name="role",
                native_enum=False,
                create_constraint=True,
                length=32,
            ),
            nullable=False,
        ),
        sa.Column("model", sa.String(), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("text", sa.String(), nullable=False),
        sa.Column(
            "status",
            sa.Enum(
                "draft",
                "candidate",
                "active",
                "retired",
                name="status",
                native_enum=False,
                create_constraint=True,
                length=32,
            ),
            server_default="draft",
            nullable=False,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_prompt_versions")),
        sa.UniqueConstraint(
            "role", "model", "version", name=op.f("uq_prompt_versions_role_model_version")
        ),
    )
    op.create_table(
        "settings",
        sa.Column("key", sa.String(), nullable=False),
        sa.Column("value", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("key", name=op.f("pk_settings")),
    )
    op.create_table(
        "users",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("tailscale_login", sa.String(), nullable=False),
        sa.Column("display_name", sa.String(), nullable=True),
        sa.Column(
            "role",
            sa.Enum(
                "user", "admin", name="role", native_enum=False, create_constraint=True, length=32
            ),
            server_default="user",
            nullable=False,
        ),
        sa.Column(
            "status",
            sa.Enum(
                "pending",
                "approved",
                "blocked",
                name="status",
                native_enum=False,
                create_constraint=True,
                length=32,
            ),
            server_default="pending",
            nullable=False,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("approved_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_users")),
        sa.UniqueConstraint("tailscale_login", name=op.f("uq_users_tailscale_login")),
    )
    op.create_table(
        "workflow_versions",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("api_json", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("manifest", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("sha256", sa.String(), nullable=False),
        sa.Column(
            "status",
            sa.Enum(
                "draft",
                "candidate",
                "active",
                "retired",
                name="status",
                native_enum=False,
                create_constraint=True,
                length=32,
            ),
            server_default="draft",
            nullable=False,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_workflow_versions")),
        sa.UniqueConstraint("name", "version", name=op.f("uq_workflow_versions_name_version")),
    )
    op.create_table(
        "engines",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column(
            "slot",
            sa.Enum(
                "keyframe",
                "clip",
                "critic",
                name="slot",
                native_enum=False,
                create_constraint=True,
                length=32,
            ),
            nullable=False,
        ),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("workflow_version_id", sa.Uuid(), nullable=True),
        sa.Column("prompt_version_id", sa.Uuid(), nullable=True),
        sa.Column(
            "limits",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
        sa.Column(
            "status",
            sa.Enum(
                "draft",
                "candidate",
                "active",
                "retired",
                name="status",
                native_enum=False,
                create_constraint=True,
                length=32,
            ),
            server_default="draft",
            nullable=False,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["prompt_version_id"], ["prompt_versions.id"], name=op.f("fk_engines_prompt_version_id")
        ),
        sa.ForeignKeyConstraint(
            ["workflow_version_id"],
            ["workflow_versions.id"],
            name=op.f("fk_engines_workflow_version_id"),
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_engines")),
        sa.UniqueConstraint("slot", "name", "version", name=op.f("uq_engines_slot_name_version")),
    )
    op.create_table(
        "recipes",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column(
            "regime",
            sa.Enum(
                "multishot",
                "keyframes",
                name="regime",
                native_enum=False,
                create_constraint=True,
                length=32,
            ),
            nullable=False,
        ),
        sa.Column("keyframe_engine_id", sa.Uuid(), nullable=True),
        sa.Column("clip_engine_id", sa.Uuid(), nullable=False),
        sa.Column(
            "status",
            sa.Enum(
                "draft",
                "candidate",
                "active",
                "retired",
                name="status",
                native_enum=False,
                create_constraint=True,
                length=32,
            ),
            server_default="draft",
            nullable=False,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["clip_engine_id"], ["engines.id"], name=op.f("fk_recipes_clip_engine_id")
        ),
        sa.ForeignKeyConstraint(
            ["keyframe_engine_id"], ["engines.id"], name=op.f("fk_recipes_keyframe_engine_id")
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_recipes")),
        sa.UniqueConstraint("name", "version", name=op.f("uq_recipes_name_version")),
    )
    op.create_table(
        "jobs",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("owner_id", sa.Uuid(), nullable=False),
        sa.Column("idea_text", sa.String(), nullable=True),
        sa.Column("photo_count", sa.Integer(), nullable=False),
        sa.Column(
            "length",
            sa.Enum(
                "short", "long", name="length", native_enum=False, create_constraint=True, length=32
            ),
            nullable=False,
        ),
        sa.Column(
            "quality",
            sa.Enum(
                "hd720",
                "fhd1080",
                name="quality",
                native_enum=False,
                create_constraint=True,
                length=32,
            ),
            nullable=False,
        ),
        sa.Column(
            "lane",
            sa.Enum(
                "idea", "render", name="lane", native_enum=False, create_constraint=True, length=32
            ),
            server_default="idea",
            nullable=False,
        ),
        sa.Column(
            "status",
            sa.Enum(
                "idea_queued",
                "idea_running",
                "awaiting_approval",
                "closed",
                "render_queued",
                "rendering",
                "preview_ready",
                "expired",
                "release_queued",
                "releasing",
                "released",
                "failed",
                "cancelled",
                name="status",
                native_enum=False,
                create_constraint=True,
                length=32,
            ),
            server_default="idea_queued",
            nullable=False,
        ),
        sa.Column(
            "regime",
            sa.Enum(
                "multishot",
                "keyframes",
                name="regime",
                native_enum=False,
                create_constraint=True,
                length=32,
            ),
            nullable=True,
        ),
        sa.Column("recipe_id", sa.Uuid(), nullable=True),
        sa.Column("critic_engine_id", sa.Uuid(), nullable=True),
        sa.Column("progress", sa.Double(), server_default=sa.text("0"), nullable=False),
        sa.Column("error", sa.String(), nullable=True),
        sa.Column(
            "queued_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False
        ),
        sa.Column("claimed_by", sa.String(), nullable=True),
        sa.Column("claimed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint("photo_count BETWEEN 1 AND 3", name=op.f("ck_jobs_photo_count")),
        sa.CheckConstraint("progress BETWEEN 0 AND 1", name=op.f("ck_jobs_progress")),
        sa.ForeignKeyConstraint(
            ["critic_engine_id"], ["engines.id"], name=op.f("fk_jobs_critic_engine_id")
        ),
        sa.ForeignKeyConstraint(
            ["owner_id"], ["users.id"], name=op.f("fk_jobs_owner_id"), ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(["recipe_id"], ["recipes.id"], name=op.f("fk_jobs_recipe_id")),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_jobs")),
    )
    op.create_index("ix_jobs_owner_id_created_at", "jobs", ["owner_id", "created_at"], unique=False)
    op.create_index("ix_jobs_queue", "jobs", ["lane", "status", "queued_at"], unique=False)
    op.create_table(
        "ideas",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("job_id", sa.Uuid(), nullable=False),
        sa.Column("text", sa.String(), nullable=False),
        sa.Column("score", sa.Integer(), nullable=False),
        sa.Column("reason", sa.String(), nullable=False),
        sa.Column(
            "proposed_by",
            sa.Enum(
                "user",
                "gemma",
                name="proposed_by",
                native_enum=False,
                create_constraint=True,
                length=32,
            ),
            nullable=False,
        ),
        sa.Column("prompt_version_id", sa.Uuid(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("approved_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint("score BETWEEN 1 AND 10", name=op.f("ck_ideas_score")),
        sa.ForeignKeyConstraint(
            ["job_id"], ["jobs.id"], name=op.f("fk_ideas_job_id"), ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(
            ["prompt_version_id"], ["prompt_versions.id"], name=op.f("fk_ideas_prompt_version_id")
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_ideas")),
    )
    op.create_index(op.f("ix_ideas_job_id"), "ideas", ["job_id"], unique=False)
    op.create_table(
        "job_steps",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("job_id", sa.Uuid(), nullable=False),
        sa.Column(
            "step",
            sa.Enum(
                "idea",
                "scenario",
                "keyframe",
                "clip",
                "check",
                "join",
                "release",
                name="step",
                native_enum=False,
                create_constraint=True,
                length=32,
            ),
            nullable=False,
        ),
        sa.Column("ordinal", sa.Integer(), server_default=sa.text("0"), nullable=False),
        sa.Column("attempt", sa.Integer(), server_default=sa.text("1"), nullable=False),
        sa.Column(
            "status",
            sa.Enum(
                "pending",
                "running",
                "done",
                "failed",
                name="status",
                native_enum=False,
                create_constraint=True,
                length=32,
            ),
            server_default="pending",
            nullable=False,
        ),
        sa.Column("inputs_hash", sa.String(), nullable=True),
        sa.Column("inputs", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("output", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("workflow_version_id", sa.Uuid(), nullable=True),
        sa.Column("prompt_version_id", sa.Uuid(), nullable=True),
        sa.Column("progress", sa.Double(), server_default=sa.text("0"), nullable=False),
        sa.Column("error", sa.String(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint("attempt >= 1", name=op.f("ck_job_steps_attempt")),
        sa.CheckConstraint("progress BETWEEN 0 AND 1", name=op.f("ck_job_steps_progress")),
        sa.ForeignKeyConstraint(
            ["job_id"], ["jobs.id"], name=op.f("fk_job_steps_job_id"), ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(
            ["prompt_version_id"],
            ["prompt_versions.id"],
            name=op.f("fk_job_steps_prompt_version_id"),
        ),
        sa.ForeignKeyConstraint(
            ["workflow_version_id"],
            ["workflow_versions.id"],
            name=op.f("fk_job_steps_workflow_version_id"),
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_job_steps")),
        sa.UniqueConstraint(
            "job_id",
            "step",
            "ordinal",
            "attempt",
            name=op.f("uq_job_steps_job_id_step_ordinal_attempt"),
        ),
    )
    op.create_table(
        "assets",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("owner_id", sa.Uuid(), nullable=False),
        sa.Column("job_id", sa.Uuid(), nullable=False),
        sa.Column("step_id", sa.Uuid(), nullable=True),
        sa.Column(
            "kind",
            sa.Enum(
                "photo",
                "gemma_copy",
                "keyframe",
                "latent",
                "clip",
                "preview",
                "release",
                name="kind",
                native_enum=False,
                create_constraint=True,
                length=32,
            ),
            nullable=False,
        ),
        sa.Column(
            "store",
            sa.Enum(
                "shared",
                "private",
                name="store",
                native_enum=False,
                create_constraint=True,
                length=32,
            ),
            nullable=False,
        ),
        sa.Column("path", sa.String(), nullable=False),
        sa.Column("mime", sa.String(), nullable=False),
        sa.Column("size_bytes", sa.BigInteger(), nullable=False),
        sa.Column("sha256", sa.String(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(
            ["job_id"], ["jobs.id"], name=op.f("fk_assets_job_id"), ondelete="RESTRICT"
        ),
        sa.ForeignKeyConstraint(
            ["owner_id"], ["users.id"], name=op.f("fk_assets_owner_id"), ondelete="RESTRICT"
        ),
        sa.ForeignKeyConstraint(
            ["step_id"], ["job_steps.id"], name=op.f("fk_assets_step_id"), ondelete="SET NULL"
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_assets")),
        sa.UniqueConstraint("store", "path", name=op.f("uq_assets_store_path")),
    )
    op.create_index(
        "ix_assets_expires_at",
        "assets",
        ["expires_at"],
        unique=False,
        postgresql_where=sa.text("expires_at IS NOT NULL"),
    )
    op.create_index(op.f("ix_assets_job_id"), "assets", ["job_id"], unique=False)
    op.create_index(op.f("ix_assets_owner_id"), "assets", ["owner_id"], unique=False)

    # Nothing runs paused by default; no recipe is active until one is promoted (E2).
    op.execute(
        "INSERT INTO settings (key, value) VALUES "
        "('paused', 'false'), ('active_recipes', '{}'), ('active_critic_engine', 'null')"
    )


def downgrade() -> None:
    op.drop_table("assets")
    op.drop_table("job_steps")
    op.drop_table("ideas")
    op.drop_table("jobs")
    op.drop_table("recipes")
    op.drop_table("engines")
    op.drop_table("workflow_versions")
    op.drop_table("users")
    op.drop_table("settings")
    op.drop_table("prompt_versions")
    op.drop_table("file_deletions")
