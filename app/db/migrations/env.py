"""Alembic migration environment.

Two edits from the generated template, both deliberate:

1. The database URL comes from `app.config.get_settings()`, NOT from alembic.ini.
   Reason: the password would otherwise be committed to git, and the app and the
   migrations could drift onto different databases. One source of truth.

2. `target_metadata` points at our real Base.metadata so `alembic revision
   --autogenerate` can diff the models against the live schema.
"""

from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

from app.config import get_settings
from app.db.models import Base  # noqa: F401 — import registers models in Base.metadata

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Inject the URL at runtime, overriding whatever alembic.ini says.
config.set_main_option("sqlalchemy.url", get_settings().database_url)

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """Emit SQL to stdout instead of executing it (`alembic upgrade head --sql`).

    Useful when a DBA has to review or apply the SQL by hand.
    """
    context.configure(
        url=config.get_main_option("sqlalchemy.url"),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Connect to the database and apply migrations.

    NullPool: this is a short-lived one-shot script, so connection pooling would only
    delay process exit.
    """
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,  # detect column type changes, not just added/dropped cols
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
