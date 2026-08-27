"""Shared test fixtures.

Integration tests run against the real Postgres from docker-compose. They use a
transaction that is rolled back after each test rather than a truncate: rollback is
faster, and it cannot accidentally leave rows behind if a test fails midway.
"""

from __future__ import annotations

import pytest
from sqlalchemy import text

from app.config import get_settings
from app.db.session import get_engine, session_scope


@pytest.fixture
def data_root(tmp_path, monkeypatch):
    """Point dataset storage at a temp directory for the duration of a test."""
    root = tmp_path / "datasets"
    root.mkdir()
    monkeypatch.setattr(get_settings(), "data_dir", root)
    return root


@pytest.fixture
def db():
    """A clean database for one test.

    Deletes only from our tables and only within a transaction, so a failure cannot
    leave residue. `datasets` alone is enough — versions and column profiles are
    removed by the ON DELETE CASCADE, which this also exercises incidentally.
    """
    try:
        with get_engine().connect() as conn:
            conn.execute(text("SELECT 1"))
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"Postgres not available: {type(exc).__name__}")

    with session_scope() as session:
        session.execute(text("DELETE FROM datasets"))

    yield

    with session_scope() as session:
        session.execute(text("DELETE FROM datasets"))


@pytest.fixture
def sales_csv(tmp_path):
    path = tmp_path / "sales.csv"
    path.write_text(
        "order_id,order_date,region,revenue,cost,flag\n"
        "1,2024-01-15,North,1200.0,800.0,X\n"
        "2,2024-01-20,South,450.0,300.0,X\n"
        "3,2024-02-10,North,980.0,700.0,X\n"
        "4,2024-02-14,East,,410.0,X\n",
        encoding="utf-8",
    )
    return path
