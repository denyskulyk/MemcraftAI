from logging.config import fileConfig

from alembic import context
from sqlalchemy import create_engine, pool

from memcraft.config import get_settings
from memcraft.models import Base

if context.config.config_file_name is not None:
    fileConfig(context.config.config_file_name, disable_existing_loggers=False)

# A test may pass its own address with `config.set_main_option("sqlalchemy.url", ...)`.
url = context.config.get_main_option("sqlalchemy.url") or get_settings().database_url


def run_offline() -> None:
    context.configure(
        url=url,
        target_metadata=Base.metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_online() -> None:
    engine = create_engine(url, poolclass=pool.NullPool, connect_args={"connect_timeout": 5})
    with engine.connect() as connection:
        context.configure(connection=connection, target_metadata=Base.metadata, compare_type=True)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_offline()
else:
    run_online()
