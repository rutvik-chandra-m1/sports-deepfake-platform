"""
Alembic environment (R11).

Reads the database URL from the application's own Settings rather than
alembic.ini, so migrations always target the same database the app does --
including the path-anchoring fix that makes relative .env paths resolve
against backend/ instead of the current working directory.
"""

import sys
from logging.config import fileConfig
from pathlib import Path

from sqlalchemy import engine_from_config, pool

from alembic import context

# backend/alembic/env.py -> backend/
BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from app.core.config import get_settings  # noqa: E402
from app.db.base import Base  # noqa: E402
from app.models import analysis  # noqa: E402,F401 -- registers tables on Base.metadata

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# The app is the single source of truth for where the database lives.
# '%' is escaped because ConfigParser treats it as interpolation syntax.
config.set_main_option("sqlalchemy.url", get_settings().database_url.replace("%", "%%"))

target_metadata = Base.metadata


def _include_object(object, name, type_, reflected, compare_to):  # noqa: A002, ANN001
    """Keep autogenerate focused on application tables."""
    if type_ == "table" and name == "alembic_version":
        return False
    return True


# SQLite cannot ALTER/DROP COLUMN. Batch mode rebuilds the table instead;
# without it, any column alteration generates SQL that SQLite rejects.
_COMMON = dict(
    target_metadata=target_metadata,
    compare_type=True,
    include_object=_include_object,
    render_as_batch=True,
)


def run_migrations_offline() -> None:
    context.configure(
        url=config.get_main_option("sqlalchemy.url"),
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        **_COMMON,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(connection=connection, **_COMMON)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
