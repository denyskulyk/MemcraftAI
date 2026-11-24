"""The migration and the models must describe the same schema.

Without a database the two are compared as SQL text; `postgres` tests run it for real.
"""

import io
import os
import re

import pytest
from alembic import command
from alembic.autogenerate import compare_metadata
from alembic.config import Config
from alembic.migration import MigrationContext
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.dialects import postgresql
from sqlalchemy.engine import make_url
from sqlalchemy.schema import CreateIndex, CreateTable

from memcraft.models import Base

BACKLOG_TABLES = {
    "users",
    "jobs",
    "job_steps",
    "ideas",
    "assets",
    "workflow_versions",
    "prompt_versions",
    "engines",
    "recipes",
    "settings",
}


def alembic_config(url: str, output: io.StringIO | None = None) -> Config:
    root = os.path.dirname(os.path.dirname(__file__))
    config = Config(os.path.join(root, "alembic.ini"), output_buffer=output)
    config.set_main_option("script_location", os.path.join(root, "migrations"))
    config.set_main_option("sqlalchemy.url", url.replace("%", "%%"))
    return config


def statements(sql: str) -> set[tuple[str, ...]]:
    """Each statement as its sorted lines, so the order of constraints does not matter."""
    result = set()
    for statement in re.split(r";\s*(?:\n|$)", sql):
        lines = [re.sub(r"\s+", " ", x).strip(" ,") for x in statement.splitlines()]
        if any(lines):
            result.add(tuple(sorted(x for x in lines if x and not x.startswith("--"))))
    return result


def test_backlog_tables_exist():
    assert set(Base.metadata.tables) >= BACKLOG_TABLES


def test_migration_matches_models():
    output = io.StringIO()
    command.upgrade(alembic_config("postgresql+psycopg://x@x/x", output), "head", sql=True)
    sql = output.getvalue()
    migrated = {
        s
        for s in statements(sql)
        if any(x.startswith(("CREATE TABLE", "CREATE INDEX", "CREATE UNIQUE INDEX")) for x in s)
        and not any("alembic_version" in x for x in s)
    }
    # A column added later belongs to the statement that created its table.
    for table, column in re.findall(r"ALTER TABLE (\w+) ADD COLUMN (.+);", sql):
        [created] = [s for s in migrated if f"CREATE TABLE {table} (" in s]
        migrated.remove(created)
        migrated.add(tuple(sorted((*created, column))))
    # A column made optional later loses its NOT NULL in the statement that created it.
    for table, column in re.findall(r"ALTER TABLE (\w+) ALTER COLUMN (\w+) DROP NOT NULL;", sql):
        [created] = [s for s in migrated if f"CREATE TABLE {table} (" in s]
        migrated.remove(created)
        migrated.add(
            tuple(
                sorted(
                    x.removesuffix(" NOT NULL") if x.startswith(f"{column} ") else x
                    for x in created
                )
            )
        )
    dialect = postgresql.dialect()
    ddl = []
    for table in Base.metadata.sorted_tables:
        ddl.append(str(CreateTable(table).compile(dialect=dialect)))
        ddl += [str(CreateIndex(index).compile(dialect=dialect)) for index in table.indexes]
    assert migrated == statements(";\n".join(ddl))


@pytest.mark.postgres
def test_upgrade_and_downgrade_on_postgres():
    url = os.environ.get("MEMCRAFT_TEST_DATABASE_URL")
    if not url:
        pytest.skip("MEMCRAFT_TEST_DATABASE_URL is not set")
    if not (make_url(url).database or "").endswith("_test"):
        pytest.fail("refusing to touch a database whose name does not end in _test")
    config = alembic_config(url)
    engine = create_engine(url)
    command.downgrade(config, "base")
    command.upgrade(config, "head")
    try:
        with engine.connect() as connection:
            context = MigrationContext.configure(connection, opts={"compare_type": True})
            assert compare_metadata(context, Base.metadata) == []
            rows = connection.execute(text("SELECT key FROM settings")).scalars()
            assert set(rows) == {"paused", "active_recipes", "active_critic_engine"}
    finally:
        command.downgrade(config, "base")
    assert set(inspect(engine).get_table_names()) <= {"alembic_version"}
    engine.dispose()
