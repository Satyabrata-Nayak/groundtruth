"""Shared dependencies.

WHY THE ENDPOINTS ARE `def` AND NOT `async def`
-----------------------------------------------
Everything below the API blocks: SQLAlchemy's sync engine, DuckDB, and file I/O. An
`async def` endpoint runs on the event loop, so one blocking call inside it stalls
*every* concurrent request in the process — the classic way an async framework ends up
slower than a threaded one.

A plain `def` endpoint is run by Starlette in a threadpool instead, where blocking is
exactly what threads are for. This is the same reasoning that kept the database layer
and the Alembic environment synchronous (D-005): async is worth its complexity when
there is real concurrency to overlap, and here there is not.
"""

from __future__ import annotations

from collections.abc import Iterator

from sqlalchemy.orm import Session

from app.db.session import session_scope


def get_session() -> Iterator[Session]:
    """One transaction per request, committed on success and rolled back on error.

    FastAPI runs the part after `yield` when the response is finished, so the commit
    happens after the handler returns but before the client is answered — a failure
    while committing still becomes a 500 rather than a success the database refused.
    """
    with session_scope() as session:
        yield session
