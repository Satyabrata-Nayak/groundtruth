"""Declarative base for all ORM models.

Every table in the project inherits from `Base`. That matters for migrations:
`Base.metadata` is the in-code description of what the schema *should* look like, and
Alembic autogenerate works by diffing it against what the live database *actually*
looks like. If a model is never imported, it is not in the metadata, and Alembic will
silently generate a migration that drops its table. So: models get imported in
`app/db/models.py`, and that module gets imported by the migration environment.
"""

from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass
