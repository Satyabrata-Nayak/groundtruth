"""ORM models.

This module is the single import point for models: `app/db/migrations/env.py` imports
it so everything lands in `Base.metadata` before autogenerate runs. A model class that
is never imported is absent from the metadata, and autogenerate concludes its table is
unwanted and emits a `drop_table`.

WHAT LIVES HERE AND WHAT DOES NOT
---------------------------------
Postgres holds *metadata about* datasets. The dataset rows themselves live on disk as
Parquet and are read by DuckDB. Postgres is row-oriented and would be both slow and
enormous for analytical data; Parquet is columnar and DuckDB reads it without loading
it into memory.

    Postgres   dataset identity, version history, per-column statistics
    Disk       data/datasets/<id>/v<n>/data.parquet
"""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import StrEnum

from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base

__all__ = [
    "Analysis",
    "AnalysisEvent",
    "AnalysisStatus",
    "Base",
    "ColumnProfile",
    "Dataset",
    "DatasetVersion",
    "EventKind",
    "TERMINAL_STATUSES",
]


class Dataset(Base):
    """A logical dataset. Stable across re-uploads; versions carry the actual data."""

    __tablename__ = "datasets"

    id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    versions: Mapped[list[DatasetVersion]] = relationship(
        back_populates="dataset",
        cascade="all, delete-orphan",
        order_by="DatasetVersion.version",
    )


class DatasetVersion(Base):
    """One immutable snapshot of a dataset, and the profile computed from it.

    Nothing here is ever updated after ingestion completes. That is what lets a stored
    analysis reference (dataset_id, version) and be re-run to the same numbers later.
    """

    __tablename__ = "dataset_versions"
    __table_args__ = (
        # Two uploads must never claim the same version number. The filesystem already
        # enforces this via mkdir(exist_ok=False); this is the same invariant expressed
        # where the database can also enforce it.
        UniqueConstraint("dataset_id", "version", name="uq_dataset_version"),
        Index("ix_dataset_versions_dataset_id", "dataset_id"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    dataset_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False
    )
    version: Mapped[int] = mapped_column(Integer, nullable=False)

    # --- provenance: what was uploaded ---
    original_filename: Mapped[str] = mapped_column(String(512), nullable=False)
    original_format: Mapped[str] = mapped_column(String(16), nullable=False)  # csv | parquet
    source_bytes: Mapped[int] = mapped_column(BigInteger, nullable=False)

    # --- what was stored ---
    parquet_bytes: Mapped[int] = mapped_column(BigInteger, nullable=False)
    row_count: Mapped[int] = mapped_column(BigInteger, nullable=False)
    column_count: Mapped[int] = mapped_column(Integer, nullable=False)

    # --- dataset-level quality signals ---
    # Surfaced to the user, and later read by the agent BEFORE it answers, so it can
    # say "this excludes 0.03% of rows with invalid dates" instead of silently
    # averaging over nulls.
    duplicate_row_count: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)
    ingested_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    dataset: Mapped[Dataset] = relationship(back_populates="versions")
    columns: Mapped[list[ColumnProfile]] = relationship(
        back_populates="version",
        cascade="all, delete-orphan",
        order_by="ColumnProfile.position",
    )


class ColumnProfile(Base):
    """Statistics for one column of one dataset version.

    Numeric stats are nullable because they are meaningless for text and boolean
    columns. Storing NULL rather than 0 keeps "not applicable" distinguishable from
    "genuinely zero" — a distinction that matters as soon as anything reads these.
    """

    __tablename__ = "column_profiles"
    __table_args__ = (
        UniqueConstraint("version_id", "name", name="uq_column_profile_name"),
        Index("ix_column_profiles_version_id", "version_id"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    version_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("dataset_versions.id", ondelete="CASCADE"),
        nullable=False,
    )

    name: Mapped[str] = mapped_column(String(255), nullable=False)
    position: Mapped[int] = mapped_column(Integer, nullable=False)  # column order
    duckdb_type: Mapped[str] = mapped_column(String(64), nullable=False)
    # Coarse bucket used for tool routing and chart validation later.
    semantic_type: Mapped[str] = mapped_column(String(16), nullable=False)

    null_count: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)
    null_fraction: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    distinct_count: Mapped[int | None] = mapped_column(BigInteger)

    min_value: Mapped[str | None] = mapped_column(Text)
    max_value: Mapped[str | None] = mapped_column(Text)
    mean_value: Mapped[float | None] = mapped_column(Float)
    stddev_value: Mapped[float | None] = mapped_column(Float)
    q25_value: Mapped[float | None] = mapped_column(Float)
    q50_value: Mapped[float | None] = mapped_column(Float)
    q75_value: Mapped[float | None] = mapped_column(Float)

    # --- quality flags, computed at ingest ---
    is_constant: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    is_high_cardinality: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    # `back_populates` names the attribute on the OTHER class and requires both sides to
    # exist. DatasetVersion.columns pointed here with back_populates="version" while this
    # side was missing, which SQLAlchemy reports only at first mapper use — not at import,
    # and not by any linter.
    version: Mapped[DatasetVersion] = relationship(back_populates="columns")


# ================================================================================
# M4 — the job queue
# ================================================================================
#
# WHY A JOB LIVES IN POSTGRES AND NOT IN THE WEB PROCESS
# -----------------------------------------------------
# An analysis will, in M5, spend 10-60 seconds calling a local model. Doing that
# inside the HTTP request means the browser holds a connection open for a minute,
# any reverse proxy times it out, a page refresh loses the work, and a restart
# loses every in-flight request. So the request only WRITES A ROW; a separate
# process picks it up. That row is the queue.
#
# WHY NOT CELERY / REDIS / RQ
# ---------------------------
# Those need a second stateful service, and they hand back a task id that is not
# the thing the user asked about. Here the queue row IS the analysis: the same row
# carries the question, the status the UI polls, the result, and the audit trail.
# One store, one transaction, one thing to back up. Postgres already provides the
# hard part — row-level locks and SKIP LOCKED — and the whole claim is one
# statement. See D-022.


class AnalysisStatus(StrEnum):
    """The states an analysis can be in.

    Terminal states are SUCCEEDED, FAILED and CANCELLED. A worker only ever moves a
    row out of PENDING or RUNNING; nothing moves a row out of a terminal state, which
    is what makes "the result is final" true rather than hoped for.
    """

    PENDING = "PENDING"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


TERMINAL_STATUSES = frozenset(
    {AnalysisStatus.SUCCEEDED, AnalysisStatus.FAILED, AnalysisStatus.CANCELLED}
)


class EventKind(StrEnum):
    """What kind of thing happened. Deliberately a small, closed vocabulary.

    These are the agent's observable trace. Note what is absent: there is no THOUGHT
    kind. Chain-of-thought is not persisted — it is unverifiable narration, and storing
    it would invite the UI to present a model's self-description as evidence. Only
    things that actually happened get a row.

    MODEL_CALL is the borderline case, and it is here because a round trip to the model
    IS a thing that happened: it has a duration, a token count and a decision that came
    out of it. Its payload records how long it took and what it chose to do, never what
    it said to itself on the way there.
    """

    QUEUED = "QUEUED"
    CLAIMED = "CLAIMED"
    RECLAIMED = "RECLAIMED"
    MODEL_CALL = "MODEL_CALL"
    TOOL_CALL = "TOOL_CALL"
    TOOL_RESULT = "TOOL_RESULT"
    NOTE = "NOTE"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class Analysis(Base):
    """One question asked of one dataset version: the queue row and the answer.

    WHY THE VERSION IS PINNED HERE
    ------------------------------
    `dataset_version` is stored, not looked up at run time. A dataset can gain a new
    version between the question being asked and the worker picking it up, and an
    answer computed against v2 while the user was looking at v1 is wrong in the worst
    way — it is plausible. Pinning also makes the analysis re-runnable to the same
    numbers later, which is the entire reason M2 made versions immutable.
    """

    __tablename__ = "analyses"
    __table_args__ = (
        # Status is a CHECK-constrained string rather than a native Postgres ENUM.
        # Native enums are genuinely awkward to evolve: ALTER TYPE ... ADD VALUE cannot
        # run in the same transaction that then uses the new value, so a migration that
        # adds a state and backfills rows with it needs two migrations. A CHECK
        # constraint is one ALTER TABLE and gives the same guarantee.
        CheckConstraint(
            "status IN ('PENDING', 'RUNNING', 'SUCCEEDED', 'FAILED', 'CANCELLED')",
            name="ck_analyses_status",
        ),
        # THE CLAIM INDEX. The worker's claim query filters status='PENDING' and orders
        # by created_at. A partial index holds only pending rows, so it stays small
        # forever no matter how many finished analyses accumulate behind it — what a
        # worker scans is proportional to the backlog, not to history.
        Index(
            "ix_analyses_pending",
            "created_at",
            postgresql_where=text("status = 'PENDING'"),
        ),
        # THE RECLAIM INDEX, by the same argument, for the sweep that finds analyses
        # whose worker died.
        Index(
            "ix_analyses_running_heartbeat",
            "heartbeat_at",
            postgresql_where=text("status = 'RUNNING'"),
        ),
        Index("ix_analyses_dataset_id", "dataset_id"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    dataset_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False
    )
    dataset_version: Mapped[int] = mapped_column(Integer, nullable=False)
    question: Mapped[str] = mapped_column(Text, nullable=False)

    # WHICH MODEL ANSWERED, PINNED THE SAME WAY THE DATASET VERSION IS.
    #
    # Stored on the row rather than read from configuration when the worker picks it
    # up, for the same reason `dataset_version` is: the answer has to stay explicable
    # later. Two analyses of the same question can legitimately disagree because one
    # was asked of a 3B model and one of a 4B, and a result that cannot say which is a
    # result nobody can act on. NULL means "whatever the worker was configured with",
    # which is what every row written before this column existed means.
    llm_model: Mapped[str | None] = mapped_column(String(128))
    # Tri-state on purpose. True/False are an explicit choice by the asker; NULL means
    # they expressed none and the model's own default applies.
    llm_thinking: Mapped[bool | None] = mapped_column(Boolean)

    # Which thread this question belongs to, and where in it. NULL is a one-off ask,
    # which is what every analysis made before conversations existed was.
    conversation_id: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("conversations.id", ondelete="SET NULL")
    )
    # Ordering within the thread. Not derived from created_at: two questions asked in
    # the same millisecond would then have no defined order, and the whole point of the
    # thread is that "the previous answer" is unambiguous.
    turn_index: Mapped[int | None] = mapped_column(Integer)

    # A caller-supplied key that makes POST /analyses safe to retry. Unique, and
    # nullable: Postgres permits many NULLs in a unique index, so callers that do not
    # care are not forced to invent one. See D-024.
    idempotency_key: Mapped[str | None] = mapped_column(String(128), unique=True)

    status: Mapped[str] = mapped_column(
        String(16), nullable=False, default=AnalysisStatus.PENDING, server_default="PENDING"
    )

    # --- claim bookkeeping ---
    # Incremented when a worker CLAIMS the row, not when one finishes it. A worker that
    # is killed mid-analysis has still consumed an attempt, which is what stops a job
    # that reliably kills its worker from being retried forever.
    attempts: Mapped[int] = mapped_column(Integer, nullable=False, default=0, server_default="0")
    worker_id: Mapped[str | None] = mapped_column(String(128))
    heartbeat_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    # Set by the API; read by the worker at each checkpoint. A flag rather than a direct
    # status write, because only the process actually running the work can stop it
    # cleanly and report where it stopped.
    cancel_requested: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, server_default="false"
    )

    # --- timing ---
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    # --- outcome ---
    # JSONB, not JSON: it is stored parsed, so it can be indexed and queried later
    # without re-parsing text on every read. The shape is the one M5 will also produce
    # — an answer, the steps that produced it, and the evidence — so replacing the
    # hardcoded analysis with a model-driven one does not change this contract.
    result: Mapped[dict | None] = mapped_column(JSONB)
    error: Mapped[str | None] = mapped_column(Text)

    events: Mapped[list[AnalysisEvent]] = relationship(
        back_populates="analysis",
        cascade="all, delete-orphan",
        order_by="AnalysisEvent.id",
    )


class AnalysisEvent(Base):
    """One observable thing that happened while an analysis ran.

    WHY THE PRIMARY KEY IS A SEQUENCE AND NOT A PER-ANALYSIS COUNTER
    ---------------------------------------------------------------
    The obvious design is `seq = max(seq) + 1` per analysis. That is a read-then-write
    race: two writers both read 4 and both write 5. It happens to be safe today because
    exactly one worker owns a running analysis — but "safe because of an invariant
    enforced somewhere else" is how races get shipped.

    A BIGINT identity column is allocated by Postgres from a sequence, is monotonic, and
    needs no coordination. It also doubles as the polling cursor: the UI asks for
    `?after=<last id it saw>` and can neither miss an event nor see one twice.
    """

    __tablename__ = "analysis_events"
    __table_args__ = (Index("ix_analysis_events_analysis_id", "analysis_id", "id"),)

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    analysis_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("analyses.id", ondelete="CASCADE"), nullable=False
    )
    kind: Mapped[str] = mapped_column(String(24), nullable=False)
    message: Mapped[str] = mapped_column(Text, nullable=False)
    payload: Mapped[dict | None] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    analysis: Mapped[Analysis] = relationship(back_populates="events")


class Conversation(Base):
    """A thread of questions about one dataset version.

    WHY A THREAD IS AN ENTITY AND NOT JUST A UI ARRAY
    -------------------------------------------------
    Without this, "what about last year?" is unanswerable: the worker sees one question,
    with no idea what "last year" is being compared to. The thread is what makes a
    follow-up mean something, and it has to live in the database rather than in React
    state because the worker — a different process — is the thing that needs it.

    PINNED TO A DATASET VERSION, LIKE AN ANALYSIS
    ---------------------------------------------
    A conversation cannot span datasets. Half a thread about `retail` and half about
    `sensors` would let the model carry a fact from one into an answer about the other,
    which is the exact class of plausible-but-wrong this project exists to prevent.
    """

    __tablename__ = "conversations"

    id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    dataset_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False
    )
    dataset_version: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class AnswerCache(Base):
    """The same question, on the same data, answered by the same model: replay it.

    WHY THIS IS WORTH A TABLE
    -------------------------
    An analysis costs 90-190 seconds of local GPU. Asking it twice costs 380. The result
    is a pure function of four things — dataset, version, question, model — and every one
    of them is already known before any work starts, so the second ask can be answered in
    about five milliseconds from a primary-key lookup.

    THE CACHE KEY IS THE CORRECTNESS ARGUMENT
    -----------------------------------------
    `dataset_version` is in the key, so a new upload cannot serve a stale answer — the
    same reason the version is pinned on the analysis itself. `llm_model` is in the key,
    because the two models genuinely disagree and a cached Qwen2.5 answer must not be
    served to somebody who asked for Qwen3. The question is normalised (lowercased,
    whitespace collapsed, trailing punctuation dropped) so "Which country earns most?"
    and "which country earns most" are one entry rather than two.

    WHAT IS DELIBERATELY NOT CACHED
    -------------------------------
    Anything that failed, and anything carrying an unverified figure. Replaying a wrong
    answer instantly is worse than recomputing it slowly, because speed reads as
    confidence.

    NEXT STEP, DESIGNED FOR AND NOT BUILT: `question_embedding vector(768)` alongside the
    hash, so "which country earns most" also hits "top country by revenue". That needs
    the pgvector extension and an embedding model; the exact-match key below is the
    subset of it that costs neither.
    """

    __tablename__ = "answer_cache"
    __table_args__ = (
        # One row per (data, question, model). The lookup is this index.
        UniqueConstraint(
            "dataset_id",
            "dataset_version",
            "question_hash",
            "llm_model",
            name="uq_answer_cache_key",
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    dataset_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False
    )
    dataset_version: Mapped[int] = mapped_column(Integer, nullable=False)
    # sha256 of the normalised question. Hashed rather than stored raw because it is an
    # index key: a 64-character fixed width beats an unbounded question in a unique index.
    question_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    # Kept alongside the hash so a human reading this table can see what was asked.
    question: Mapped[str] = mapped_column(Text, nullable=False)
    llm_model: Mapped[str] = mapped_column(String(128), nullable=False)
    result: Mapped[dict] = mapped_column(JSONB, nullable=False)
    # Evidence that the cache is worth having, rather than a belief that it is.
    hit_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0, server_default="0")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
