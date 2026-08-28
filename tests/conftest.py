"""Shared test fixtures.

Integration tests run against the real Postgres from docker-compose. They use a
transaction that is rolled back after each test rather than a truncate: rollback is
faster, and it cannot accidentally leave rows behind if a test fails midway.

THE SUITE RUNS AGAINST ITS OWN DATABASE
---------------------------------------
`POSTGRES_DB` is redirected to `adi_test` at the top of this file, created and migrated
automatically. The `db` fixture empties tables, and pointing that at the database the
app is actually using is how a test run silently destroys a real upload.

THE SUITE RUNS WITH NO LANGUAGE MODEL, BY DEFAULT AND BY FORCE
--------------------------------------------------------------
`fixed_analysis_engine` is autouse, so every test that reaches `run_analysis` gets the
deterministic engine unless it deliberately asks otherwise. Without it the default
config (`ANALYSIS_ENGINE=agent`) would point the worker tests at Ollama, and a suite
whose result depends on which model is pulled on the machine running it is not a test
suite — it is a mood.

The agent itself is tested against a scripted model in `tests/test_agent.py`, which
needs no server at all.
"""

from __future__ import annotations

import os

# BEFORE app.config is imported, and therefore before any engine exists.
#
# The `db` fixture clears the tables it uses. Pointed at the development database that
# is also the one the running app uses, that is a data-loss bug wearing a test fixture:
# it deleted a real 542,000-row upload twice in one afternoon, and the only symptom was
# the API answering "no dataset <uuid>" some time later.
#
# An environment variable set here wins over the .env file (pydantic-settings reads the
# environment first), and alembic's env.py reads the same `get_settings()`, so this one
# line redirects the engine, the migrations and the app consistently.
os.environ["POSTGRES_DB"] = os.environ.get("TEST_POSTGRES_DB", "adi_test")

import pytest  # noqa: E402
import sqlalchemy  # noqa: E402
from sqlalchemy import text  # noqa: E402

from app.config import get_settings  # noqa: E402
from app.db.session import get_engine, session_scope  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def test_database() -> None:
    """Create the test database if it does not exist, and migrate it to head.

    Done once per session and in code rather than in a README step, because a setup
    instruction that can be skipped is one that eventually is — and the failure mode of
    skipping it is running the suite against real data.
    """
    settings = get_settings()
    admin_url = settings.database_url.rsplit("/", 1)[0] + "/postgres"
    try:
        admin = sqlalchemy.create_engine(
            admin_url, isolation_level="AUTOCOMMIT", connect_args={"connect_timeout": 5}
        )
        with admin.connect() as conn:
            exists = conn.scalar(
                text("SELECT 1 FROM pg_database WHERE datname = :name"),
                {"name": settings.postgres_db},
            )
            if not exists:
                # CREATE DATABASE cannot run inside a transaction, hence AUTOCOMMIT.
                conn.execute(text(f'CREATE DATABASE "{settings.postgres_db}"'))
        admin.dispose()
    except Exception as exc:  # noqa: BLE001 - no server is a skip, not an error
        pytest.skip(f"Postgres not available: {type(exc).__name__}")

    from alembic import command
    from alembic.config import Config

    command.upgrade(Config("alembic.ini"), "head")


@pytest.fixture(autouse=True)
def fixed_analysis_engine(monkeypatch):
    """No test calls a language model unless it says so explicitly."""
    monkeypatch.setattr(get_settings(), "analysis_engine", "fixed")


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
