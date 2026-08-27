"""Dataset lifecycle: the operations the API and worker actually call.

This module is where the pieces built separately in M2 become one thing:

    upload ──► ingest ──► profile ──► persist ──► a dataset you can query later

WHY THIS LAYER EXISTS AT ALL
----------------------------
`ingest_file` and `profile_parquet` both work in isolation, but a caller that used them
directly would have to remember to profile after ingesting, to persist both, to clean
up the Parquet file if the database write failed, and to keep the two consistent. That
is business logic, and leaving it to every caller means each caller gets it subtly
wrong in a different way.

THE CONSISTENCY PROBLEM, AND HOW IT IS HANDLED
----------------------------------------------
There are two stores here and they cannot be committed atomically:

    filesystem   data/datasets/<id>/v1/data.parquet     (no transactions)
    Postgres     datasets / dataset_versions / columns  (transactional)

The failure that matters is a written Parquet file with no database row: invisible to
every listing, occupying disk forever, and impossible to find without walking
directories. So the order is deliberate — write the file first, then commit the
metadata, and if the commit fails, delete the file. The reverse order would risk a
database row pointing at a file that does not exist, which is worse: it appears in
listings and fails only when someone tries to query it.
"""

from __future__ import annotations

import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from app.data import ingest, storage
from app.data.profile import DatasetProfile, profile_parquet
from app.db.models import ColumnProfile, Dataset, DatasetVersion
from app.db.session import session_scope


@dataclass(frozen=True)
class CreatedDataset:
    dataset_id: uuid.UUID
    version_id: uuid.UUID
    version: int
    name: str
    row_count: int
    column_count: int


def create_dataset(
    source: Path,
    *,
    name: str | None = None,
    description: str | None = None,
    dataset_id: uuid.UUID | str | None = None,
    original_filename: str | None = None,
) -> CreatedDataset:
    """Ingest, profile and register a file as a queryable dataset version.

    Passing `dataset_id` adds a version to an existing dataset; omitting it creates a
    new one. Either the whole operation succeeds or nothing is left behind.
    """
    result = ingest.ingest_file(
        source, dataset_id=dataset_id, original_filename=original_filename
    )

    try:
        profile = profile_parquet(result.parquet_path)

        with session_scope() as session:
            dataset = _get_or_create_dataset(
                session,
                dataset_id=result.dataset_id,
                name=name or Path(result.original_filename).stem,
                description=description,
            )

            version = DatasetVersion(
                dataset_id=dataset.id,
                version=result.version,
                original_filename=result.original_filename,
                original_format=result.original_format,
                source_bytes=result.source_bytes,
                parquet_bytes=result.parquet_bytes,
                row_count=profile.row_count,
                column_count=profile.column_count,
                duplicate_row_count=profile.duplicate_row_count,
            )
            session.add(version)
            session.flush()  # assign version.id before building children

            session.add_all(
                [
                    ColumnProfile(
                        version_id=version.id,
                        name=column.name,
                        position=column.position,
                        duckdb_type=column.duckdb_type,
                        semantic_type=column.semantic_type,
                        null_count=column.null_count,
                        null_fraction=column.null_fraction,
                        distinct_count=column.distinct_count,
                        min_value=column.min_value,
                        max_value=column.max_value,
                        mean_value=column.mean_value,
                        stddev_value=column.stddev_value,
                        q25_value=column.q25_value,
                        q50_value=column.q50_value,
                        q75_value=column.q75_value,
                        is_constant=column.is_constant,
                        is_high_cardinality=column.is_high_cardinality,
                    )
                    for column in profile.columns
                ]
            )

            created = CreatedDataset(
                dataset_id=dataset.id,
                version_id=version.id,
                version=version.version,
                name=dataset.name,
                row_count=version.row_count,
                column_count=version.column_count,
            )
    except Exception:
        # The metadata never committed, so the Parquet file would be unreachable —
        # invisible to listings, occupying disk, findable only by walking directories.
        shutil.rmtree(result.parquet_path.parent, ignore_errors=True)
        raise

    return created


def _get_or_create_dataset(
    session: Session, *, dataset_id: uuid.UUID, name: str, description: str | None
) -> Dataset:
    dataset = session.get(Dataset, dataset_id)
    if dataset is None:
        dataset = Dataset(id=dataset_id, name=name, description=description)
        session.add(dataset)
        session.flush()
    return dataset


# --------------------------------------------------------------------------------
# Reads
# --------------------------------------------------------------------------------


def list_datasets(session: Session, *, limit: int = 100) -> list[Dataset]:
    """Newest first. `selectinload` fetches versions in one extra query rather than
    one per dataset — the N+1 problem, which is invisible with three datasets and
    ruinous with three hundred."""
    stmt = (
        select(Dataset)
        .options(selectinload(Dataset.versions))
        .order_by(Dataset.created_at.desc())
        .limit(limit)
    )
    return list(session.scalars(stmt))


def get_dataset(session: Session, dataset_id: uuid.UUID | str) -> Dataset | None:
    return session.get(Dataset, storage.parse_dataset_id(dataset_id))


def get_version(
    session: Session, dataset_id: uuid.UUID | str, version: int | None = None
) -> DatasetVersion | None:
    """One version's metadata and column profiles. `version=None` means the latest."""
    stmt = (
        select(DatasetVersion)
        .options(selectinload(DatasetVersion.columns))
        .where(DatasetVersion.dataset_id == storage.parse_dataset_id(dataset_id))
    )
    stmt = (
        stmt.order_by(DatasetVersion.version.desc()).limit(1)
        if version is None
        else stmt.where(DatasetVersion.version == version)
    )
    return session.scalars(stmt).first()


def get_stored_profile(
    session: Session, dataset_id: uuid.UUID | str, version: int | None = None
) -> DatasetProfile | None:
    """Rebuild the profile from Postgres instead of recomputing it.

    This is the point of persisting it: profiling a multi-million-row file takes
    seconds, and it is computed once at ingest and read back instantly forever after.
    """
    stored = get_version(session, dataset_id, version)
    if stored is None:
        return None

    from app.data.profile import ColumnStats

    return DatasetProfile(
        row_count=stored.row_count,
        column_count=stored.column_count,
        duplicate_row_count=stored.duplicate_row_count,
        columns=[
            ColumnStats(
                name=c.name,
                position=c.position,
                duckdb_type=c.duckdb_type,
                semantic_type=c.semantic_type,
                null_count=c.null_count,
                null_fraction=c.null_fraction,
                distinct_count=c.distinct_count,
                min_value=c.min_value,
                max_value=c.max_value,
                mean_value=c.mean_value,
                stddev_value=c.stddev_value,
                q25_value=c.q25_value,
                q50_value=c.q50_value,
                q75_value=c.q75_value,
                is_constant=c.is_constant,
                is_high_cardinality=c.is_high_cardinality,
            )
            for c in stored.columns
        ],
    )


def delete_dataset(dataset_id: uuid.UUID | str) -> bool:
    """Remove a dataset from both stores. Returns False if it did not exist.

    Database first: the ORM cascade removes versions and column profiles in one
    transaction. Files second, because an orphaned file is recoverable by hand while a
    row pointing at a deleted file breaks every listing that touches it.
    """
    ds_id = storage.parse_dataset_id(dataset_id)
    with session_scope() as session:
        dataset = session.get(Dataset, ds_id)
        if dataset is None:
            return False
        session.delete(dataset)

    shutil.rmtree(storage.dataset_dir(ds_id), ignore_errors=True)
    return True
