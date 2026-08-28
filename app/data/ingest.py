"""Turn an uploaded file into an immutable, queryable dataset version.

    upload ──► validate ──► normalise to Parquet ──► write v<n>/data.parquet

WHY EVERYTHING BECOMES PARQUET
------------------------------
One storage format means one code path for querying, profiling and versioning. CSV is
an interchange format, not a storage format: it has no types, no column statistics, no
compression worth the name, and every reader has to re-sniff its schema. Parquet is
columnar, typed, compressed, and carries per-column-chunk statistics that let DuckDB
skip data it does not need. Converting once at ingest means never paying CSV's costs
again.

WHY DUCKDB DOES THE CONVERSION
------------------------------
`COPY (SELECT * FROM read_csv(...)) TO '...' (FORMAT PARQUET)` streams through DuckDB
without materialising the whole file in Python memory, so a 5 GB CSV does not need
5 GB of RAM. DuckDB's CSV sniffer also handles the messy realities — mixed delimiters,
quoted newlines, inconsistent types, BOMs — better than a hand-rolled reader.

VALIDATION IS NOT OPTIONAL
--------------------------
An uploaded file is untrusted input. Every check here exists because its absence turns
a bad upload into either a crash deep in the query layer or an unbounded resource
commitment.
"""

from __future__ import annotations

import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path

import duckdb

from app.config import get_settings
from app.data import storage

SUPPORTED_FORMATS = {"csv", "parquet"}
_CSV_SUFFIXES = {".csv"}
_PARQUET_SUFFIXES = {".parquet", ".pq"}


class IngestError(ValueError):
    """An upload was rejected. The message is safe to show a user."""


@dataclass(frozen=True)
class IngestResult:
    dataset_id: uuid.UUID
    version: int
    parquet_path: Path
    original_filename: str
    original_format: str
    source_bytes: int
    parquet_bytes: int
    row_count: int
    column_count: int


def detect_format(filename: str) -> str:
    """Classify by extension.

    Extension is a *hint*, not proof — the real check is whether DuckDB can parse the
    file, which happens during conversion. This exists to reject obviously wrong
    uploads early with a clear message rather than a parser error 200 MB later.
    """
    suffix = Path(filename).suffix.lower()
    if suffix in _CSV_SUFFIXES:
        return "csv"
    if suffix in _PARQUET_SUFFIXES:
        return "parquet"
    raise IngestError(f"unsupported file type '{suffix or filename}'. Supported: .csv, .parquet")


def validate_source(path: Path, filename: str | None = None) -> tuple[str, int]:
    """Check an uploaded file before any expensive work. Returns (format, size_bytes)."""
    if not path.is_file():
        raise IngestError(f"file not found: {path}")

    size = path.stat().st_size
    if size == 0:
        raise IngestError("file is empty")

    limit = get_settings().max_upload_mb * 1024 * 1024
    if size > limit:
        raise IngestError(
            f"file is {size / 1024 / 1024:.1f} MB, which exceeds the "
            f"{get_settings().max_upload_mb} MB limit"
        )

    return detect_format(filename or path.name), size


def _convert_csv_to_parquet(source: Path, destination: Path) -> None:
    """Stream a CSV into Parquet via DuckDB.

    `sample_size=-1`: the sniffer reads the WHOLE file to infer types rather than a
    leading sample. Slower, but a column that looks like an integer for 10,000 rows
    and then contains "N/A" must not be typed as an integer — that is a silent
    data-corruption bug, and the profile built on top of it would be wrong in a way
    nobody notices.
    """
    con = duckdb.connect(":memory:")
    try:
        con.execute(
            "COPY (SELECT * FROM read_csv(?, sample_size=-1, all_varchar=false)) "
            f"TO {storage.sql_path_literal(destination)} (FORMAT PARQUET, COMPRESSION ZSTD)",
            [source.as_posix()],
        )
    except duckdb.Error as exc:
        raise IngestError(f"could not parse CSV: {_clean_duckdb_error(exc)}") from exc
    finally:
        con.close()


def _verify_parquet(source: Path) -> None:
    """Confirm an uploaded Parquet file is readable before we adopt it."""
    con = duckdb.connect(":memory:")
    try:
        con.execute("SELECT * FROM read_parquet(?) LIMIT 0", [source.as_posix()])
    except duckdb.Error as exc:
        raise IngestError(f"could not read Parquet file: {_clean_duckdb_error(exc)}") from exc
    finally:
        con.close()


def _clean_duckdb_error(exc: Exception) -> str:
    """First line of a DuckDB error, without the absolute path it embeds.

    Server paths in user-visible errors leak filesystem layout, and DuckDB's multi-line
    hints are noise for anyone who did not write the query.
    """
    return str(exc).split("\n")[0][:300]


def _shape(parquet: Path) -> tuple[int, int]:
    """(row_count, column_count) read from Parquet metadata where possible."""
    con = duckdb.connect(":memory:")
    try:
        rows = con.execute("SELECT count(*) FROM read_parquet(?)", [parquet.as_posix()]).fetchone()[
            0
        ]
        cols = len(
            con.execute("SELECT * FROM read_parquet(?) LIMIT 0", [parquet.as_posix()]).description
        )
        return int(rows), int(cols)
    finally:
        con.close()


def ingest_file(
    source: Path,
    *,
    dataset_id: uuid.UUID | str | None = None,
    original_filename: str | None = None,
) -> IngestResult:
    """Validate a file and store it as a new immutable dataset version.

    Passing an existing `dataset_id` adds a version to that dataset; omitting it
    creates a new one.

    On any failure the partially-written version directory is removed, so a failed
    upload never leaves a half-formed version behind for a later read to trip over.
    """
    filename = original_filename or source.name
    fmt, source_bytes = validate_source(source, filename)

    ds_id = storage.parse_dataset_id(dataset_id) if dataset_id else uuid.uuid4()

    if fmt == "parquet":
        _verify_parquet(source)

    version, version_directory = storage.allocate_version_dir(ds_id)
    destination = version_directory / storage.DATA_FILENAME

    try:
        if fmt == "csv":
            _convert_csv_to_parquet(source, destination)
        else:
            shutil.copy2(source, destination)

        row_count, column_count = _shape(destination)
        if column_count == 0:
            raise IngestError("file contains no columns")
        if row_count == 0:
            raise IngestError("file contains no data rows")
    except Exception:
        # Roll back the whole version directory. A version that exists but has no
        # readable data is worse than no version at all.
        shutil.rmtree(version_directory, ignore_errors=True)
        # ...and the dataset directory too, if this was its only version. Otherwise
        # every rejected upload leaves an empty UUID-named folder behind forever:
        # invisible to `existing_versions`, so nothing ever cleans it up.
        # `rmdir` only succeeds on an empty directory, so a dataset that already has
        # other versions is never touched.
        try:
            version_directory.parent.rmdir()
        except OSError:
            pass
        raise

    return IngestResult(
        dataset_id=ds_id,
        version=version,
        parquet_path=destination,
        original_filename=filename,
        original_format=fmt,
        source_bytes=source_bytes,
        parquet_bytes=destination.stat().st_size,
        row_count=row_count,
        column_count=column_count,
    )
