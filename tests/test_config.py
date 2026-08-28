"""M1 smoke tests.

Deliberately small. The point is not coverage — there is barely any code yet — but to
establish that `pytest` runs and that the two things M1 promised (config resolves, the
database is reachable and migrated) are asserted rather than assumed.
"""

import pytest
from pydantic import ValidationError
from sqlalchemy import create_engine, text

from app.config import Settings, get_settings


def test_database_url_is_assembled_from_parts():
    """The URL is built, not stored, so the password lives in exactly one place."""
    s = Settings(
        postgres_user="u",
        postgres_password="p",
        postgres_host="h",
        postgres_port=1234,
        postgres_db="d",
    )
    assert s.database_url == "postgresql+psycopg://u:p@h:1234/d"


def test_settings_are_cached():
    """get_settings() is a lazy singleton: .env is parsed once per process."""
    assert get_settings() is get_settings()


@pytest.mark.integration
def test_database_is_reachable_and_migrated():
    """Requires `docker compose up -d` and `alembic upgrade head`.

    Asserts alembic_version exists and holds exactly one row — i.e. the migration chain
    is established and unambiguous. Two rows would mean multiple heads, which silently
    breaks future upgrades.
    """
    engine = create_engine(get_settings().database_url)
    with engine.connect() as conn:
        assert conn.execute(text("SELECT 1")).scalar_one() == 1
        count = conn.execute(text("SELECT count(*) FROM alembic_version")).scalar_one()
        assert count == 1, f"expected exactly one migration head, found {count}"


# --------------------------------------------------------------- M4 settings


def test_cors_origins_parse_from_a_comma_separated_string():
    """Stored as a string, not list[str]. pydantic-settings parses complex types from
    the environment as JSON, and CORS_ORIGINS='["http://..."]' in a .env file is a trap
    that fails at import with an opaque message."""
    settings = Settings(cors_origins="http://a.test, http://b.test ,")
    assert settings.cors_origin_list == ["http://a.test", "http://b.test"]


def test_a_heartbeat_timeout_too_close_to_the_interval_is_refused():
    """The reclaim threshold must be several beats away from the beat interval.

    At one interval, a single slow beat -- a GC pause, a blocked write, a laptop
    suspending for four seconds -- lets the sweep reclaim a job that is still running,
    and the same analysis executes twice. Catching that as a config error at import
    beats catching it as a race in production.
    """
    with pytest.raises(ValidationError, match="at least 3x"):
        Settings(worker_heartbeat_interval_s=5.0, worker_heartbeat_timeout_s=6.0)


def test_the_default_heartbeat_settings_satisfy_their_own_rule():
    settings = Settings()
    assert settings.worker_heartbeat_timeout_s >= settings.worker_heartbeat_interval_s * 3
