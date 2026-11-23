import os

import pytest
from sqlalchemy import BigInteger, create_engine, event, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.pool import StaticPool
from sqlalchemy.sql.elements import TextClause

from memcraft.models import Base


@compiles(JSONB, "sqlite")
def _jsonb(element, compiler, **kw):
    return "JSON"


@compiles(BigInteger, "sqlite")
def _bigint(element, compiler, **kw):
    return "INTEGER"  # the only type SQLite autoincrements


@compiles(TextClause, "sqlite")
def _no_casts(element, compiler, **kw):
    return compiler.visit_textclause(element, **kw).replace("::jsonb", "")


def require_test_database(name):
    if not (name or "").endswith("_test"):
        raise RuntimeError(f"refusing to create and drop tables in {name!r}: not a *_test database")


def _sqlite_engine():
    engine = create_engine(
        "sqlite://", poolclass=StaticPool, connect_args={"check_same_thread": False}
    )

    # Take BEGIN away from the driver, so that savepoints and rollbacks are real.
    @event.listens_for(engine, "connect")
    def connect(connection, _record):
        connection.isolation_level = None
        connection.execute("PRAGMA foreign_keys=ON")

    @event.listens_for(engine, "begin")
    def begin(connection):
        connection.exec_driver_sql("BEGIN")

    return engine


def _postgres_engine():
    url = os.environ.get("MEMCRAFT_TEST_DATABASE_URL")
    if not url:
        pytest.skip("MEMCRAFT_TEST_DATABASE_URL is not set")
    engine = create_engine(url, connect_args={"connect_timeout": 3})
    # Ask the server: a URL can name one database and connect to another.
    with engine.connect() as connection:
        require_test_database(connection.scalar(text("SELECT current_database()")))
    return engine


@pytest.fixture(params=["sqlite", pytest.param("postgres", marks=pytest.mark.postgres)])
def engine(request):
    engine = _sqlite_engine() if request.param == "sqlite" else _postgres_engine()
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    yield engine
    Base.metadata.drop_all(engine)
    engine.dispose()
