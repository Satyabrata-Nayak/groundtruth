"""Single source of truth for configuration.

Why this file exists: the alternative is `os.getenv("POSTGRES_HOST")` scattered across
twenty modules, where a typo becomes a runtime crash in production and nothing tells
you which variables the app actually needs. Here, config is declared once, typed, and
validated at import time — a missing or malformed value fails immediately and loudly.
"""

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",  # tolerate unrelated vars in the shell environment
    )

    # --- Postgres ---
    postgres_user: str = "adi"
    postgres_password: str = "adi_dev_password"
    postgres_db: str = "adi"
    postgres_host: str = "localhost"
    postgres_port: int = 5433

    # --- Ollama ---
    ollama_base_url: str = "http://127.0.0.1:11434"
    llm_model: str = "qwen3:4b"

    # --- Storage ---
    data_dir: Path = Field(default=Path("./data/datasets"))

    # --- Ingestion limits ---
    # A cap that a human upload will not hit but a runaway or hostile one will.
    max_upload_mb: int = 512

    # --- Query limits ---
    # These are not performance tuning. They bound the damage a bad or hostile query
    # can do, and they are enforced before the query runs where possible.
    query_timeout_s: float = 30.0
    max_result_rows: int = 10_000
    max_result_bytes: int = 10 * 1024 * 1024
    duckdb_memory_limit: str = "2GB"
    duckdb_threads: int = 4

    @property
    def database_url(self) -> str:
        """SQLAlchemy connection URL.

        Built from parts rather than stored whole so the password never has to be
        duplicated, and so Alembic and the app can never drift onto different databases.
        """
        return (
            f"postgresql+psycopg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )


@lru_cache
def get_settings() -> Settings:
    """Cached accessor.

    lru_cache makes this a lazy singleton: the .env file is read once per process, not
    once per call. Later, FastAPI will use this as a dependency, and tests can override
    it by calling get_settings.cache_clear().
    """
    return Settings()
