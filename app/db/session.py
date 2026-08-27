"""Database engine and session management.

WHY A SINGLE ENGINE
-------------------
A SQLAlchemy `Engine` owns a connection pool. Creating one per request would open a
fresh TCP connection and re-authenticate every time — Postgres connections are
expensive enough that this dominates the cost of small queries. One engine per process,
created lazily, pooled.

WHY SYNC AND NOT ASYNC
----------------------
The worker that will use this (M4) is a plain synchronous process: it claims a job,
runs DuckDB — which is itself blocking — and writes a result. There is no concurrency
for async to overlap. Async here would add a layer to reason about and buy nothing.
Same reasoning that made the Alembic environment sync (D-005).
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from functools import lru_cache

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.config import get_settings


@lru_cache
def get_engine() -> Engine:
    """The process-wide engine, created on first use.

    `pool_pre_ping` issues a cheap liveness check before handing out a pooled
    connection. Without it, a connection that died while idle — Postgres restarted, a
    firewall dropped it, the container was recycled — is handed to the caller and fails
    on first use with an opaque error. This is the single most common cause of
    mysterious "server closed the connection unexpectedly" in long-running workers.
    """
    settings = get_settings()
    return create_engine(
        settings.database_url,
        pool_pre_ping=True,
        pool_size=5,
        max_overflow=5,
        future=True,
    )


@lru_cache
def get_sessionmaker() -> sessionmaker[Session]:
    return sessionmaker(bind=get_engine(), expire_on_commit=False, future=True)


@contextmanager
def session_scope() -> Iterator[Session]:
    """A transactional scope: commit on success, roll back on any exception.

    Every write path in the application goes through this. The alternative — commit
    calls scattered through business logic — reliably produces the bug where an error
    midway leaves half a change committed.
    """
    session = get_sessionmaker()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def reset_engine() -> None:
    """Drop the cached engine and sessionmaker.

    Needed by tests that point settings at a different database after the engine has
    already been built, since both accessors are lru_cached.
    """
    if get_engine.cache_info().currsize:
        get_engine().dispose()
    get_engine.cache_clear()
    get_sessionmaker.cache_clear()
