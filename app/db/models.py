"""ORM models.

Empty in M1 by design — the baseline migration should create nothing, so that the
migration chain starts from a known-empty database. Tables arrive in M2 (datasets,
dataset profiles) and M4 (analysis jobs, analysis events).

This module is the single import point for models: `app/db/migrations/env.py` imports
it so that everything lands in `Base.metadata` before autogenerate runs.
"""

from app.db.base import Base

__all__ = ["Base"]
