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

from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base

__all__ = ["Base", "Dataset", "DatasetVersion", "ColumnProfile"]


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
