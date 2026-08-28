"""Single source of truth for configuration.

Why this file exists: the alternative is `os.getenv("POSTGRES_HOST")` scattered across
twenty modules, where a typo becomes a runtime crash in production and nothing tells
you which variables the app actually needs. Here, config is declared once, typed, and
validated at import time — a missing or malformed value fails immediately and loudly.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, model_validator
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
    # Fail fast when nothing is listening, instead of waiting out the OS TCP timeout.
    db_connect_timeout_s: int = 5

    # --- Ollama ---
    ollama_base_url: str = "http://127.0.0.1:11434"
    llm_model: str = "qwen3:4b"
    # One model call. Generous because a cold model load on a laptop is 20-30 s before
    # a single token appears, and a timeout that fires during a load looks like the
    # model is broken when it is only starting.
    llm_timeout_s: float = 300.0
    # Zero, always. This is not creative writing: we want the most probable tool call,
    # and a benchmark that cannot be reproduced is a number, not a measurement.
    llm_temperature: float = 0.0
    # None means "do not send the field", so the model's own default applies. M1
    # measured `think: false` producing WORSE machine-readable output on qwen3, because
    # the model reasons regardless and, told not to, does it inside `content` where it
    # then has to be stripped with a regex. Reasoning stays on, in its own field.
    llm_think: bool | None = None
    # MEASURED, not chosen. On a 6 GB RTX 4050 running qwen3:4b:
    #
    #     ctx=4096   61.3 tok/s   3.5 GB   100% GPU
    #     ctx=8192   61.4 tok/s   4.1 GB   100% GPU
    #     ctx=16384  29.6 tok/s   5.4 GB   17% CPU / 83% GPU   <- 2.07x slower
    #
    # The KV cache at 16k pushes the model past available VRAM and Ollama silently
    # spills layers to the CPU. Nothing reports this; it just takes twice as long.
    #
    # 8192 rather than 4096 because Ollama's default of 4096 TRUNCATES SILENTLY, and a
    # real conversation here runs ~2,400 prompt tokens plus up to 2,000 generated —
    # over 4096, which would drop the head of the prompt (the schema and the grounding
    # rules) with no error. 8192 is measurably as fast and cannot truncate.
    llm_num_ctx: int = 8192
    # How long Ollama keeps the model in VRAM after a request. The default is 5
    # minutes, so a user who thinks for six between questions pays a 20-30 second cold
    # load on the next one and blames the app.
    llm_keep_alive: str = "30m"

    # --- Agent (M5) ---
    # "agent" runs the language model; "fixed" runs M4's deterministic analysis. The
    # switch exists so the whole stack can be exercised without a model — on a machine
    # with no Ollama, in CI, and while diagnosing whether a failure is the plumbing.
    analysis_engine: str = "agent"
    # An overall ceiling on model calls, as a backstop. The loop's real bound is
    # `agent_max_tool_rounds` below; this exists so no path can spin.
    agent_max_steps: int = 6
    # How many times the model may be handed the tools. MEASURED: attaching the tool
    # definitions makes it reason 5x longer (1,547 -> 7,972 characters of thinking on
    # an identical conversation), because with tools in front of it the model
    # re-deliberates "should I call something else?" every single turn. So the tools go
    # away as soon as there is something to answer from, and the answer is written by a
    # call that has no tools attached. Two rounds leaves room for one repair after a
    # failed query; the model is told to ask for everything it needs at once instead.
    agent_max_tool_rounds: int = 2
    # ...and how long it may take in total, whatever it spends the steps on. Two
    # budgets because a fast loop going nowhere and one slow call are different
    # failures. Hitting either asks the model to answer from what it has.
    agent_time_budget_s: float = 300.0

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

    # --- Tool limits (M3) ---
    # A tool result is destined for a model's context window, so its cost is measured
    # in tokens, not bytes. These caps are far tighter than the raw query caps above:
    # a 10,000-row result would blow the context and teach the model nothing that the
    # first 50 rows did not.
    max_tool_result_rows: int = 50
    max_chart_categories: int = 50

    # --- API (M4) ---
    # The Vite dev server runs on a different origin than the API, so the browser
    # applies CORS. Stored as a comma-separated string rather than list[str] because
    # pydantic-settings parses complex types from the environment as JSON, and
    # CORS_ORIGINS='["http://localhost:5173"]' in a .env file is a trap nobody enjoys.
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"

    # --- Job queue and worker (M4) ---
    # How long a worker sleeps when it finds no work. Short enough to feel instant in
    # the UI, long enough that an idle worker is not hammering Postgres.
    worker_poll_interval_s: float = 1.0
    # A running worker refreshes analyses.heartbeat_at at this interval...
    worker_heartbeat_interval_s: float = 5.0
    # ...and any RUNNING row whose heartbeat is older than this is presumed orphaned by
    # a crashed worker and is requeued. The gap between the two is the whole safety
    # margin: too tight and a GC pause steals a live job from a healthy worker.
    worker_heartbeat_timeout_s: float = 30.0
    # A job that has been claimed this many times without finishing is failed rather
    # than requeued forever. Attempts are counted at claim time, so a crash counts.
    analysis_max_attempts: int = 3

    @model_validator(mode="after")
    def _analysis_engine_is_known(self) -> Settings:
        """A typo here must not silently select an engine nobody meant.

        `ANALYSIS_ENGINE=Agent` or `=llm` would otherwise fall through to whichever
        branch the dispatcher treats as its default, and the only symptom would be
        answers that are subtly the wrong kind.
        """
        if self.analysis_engine not in ("agent", "fixed"):
            raise ValueError(
                f"analysis_engine must be 'agent' or 'fixed', got {self.analysis_engine!r}"
            )
        return self

    @property
    def cors_origin_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

    @model_validator(mode="after")
    def _heartbeat_timeout_exceeds_interval(self) -> Settings:
        """The reclaim threshold must be several beats away from the beat interval.

        If the timeout were only one interval long, a single slow beat — a GC pause, a
        blocked write, a laptop suspending for four seconds — would let another worker
        reclaim a job that is still running, and the same analysis would execute twice.
        Three intervals means three consecutive misses before anything is presumed dead.
        """
        if self.worker_heartbeat_timeout_s < self.worker_heartbeat_interval_s * 3:
            raise ValueError(
                "worker_heartbeat_timeout_s must be at least 3x "
                "worker_heartbeat_interval_s, or a healthy worker's job can be stolen "
                f"after one missed beat (got timeout={self.worker_heartbeat_timeout_s}s, "
                f"interval={self.worker_heartbeat_interval_s}s)"
            )
        return self

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
