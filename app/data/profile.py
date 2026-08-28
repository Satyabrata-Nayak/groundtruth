"""Compute a dataset profile: what is in this data, and what is wrong with it.

This is deliberately built BEFORE any AI exists. If the system cannot describe a
dataset correctly on its own, an LLM layered on top only makes the errors harder to
find.

WHY DUCKDB `SUMMARIZE`
----------------------
One statement returns column types, approximate distinct counts, null percentages,
min/max/avg/stddev and quartiles for every column. It is the reason Polars was dropped
from the MVP (D-001): the profiling job that would have justified a second dataframe
engine is a single DuckDB command.

WHAT WE ADD ON TOP
------------------
`SUMMARIZE` describes columns. It does not answer "is this dataset trustworthy?", so
we compute duplicate rows, exact null counts, constant columns and high-cardinality
columns separately. In M5 the agent reads these BEFORE answering, so it can say
"this excludes 0.03% of rows with invalid dates" rather than silently averaging over
nulls.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import duckdb

# A text column with more distinct values than this fraction of rows is closer to an
# identifier than a category: grouping by it produces one row per record, which is a
# useless aggregation. Flagged so tools and charts can avoid it.
HIGH_CARDINALITY_FRACTION = 0.9
HIGH_CARDINALITY_MIN_ROWS = 20

_NUMERIC_PREFIXES = (
    "TINYINT",
    "SMALLINT",
    "INTEGER",
    "BIGINT",
    "HUGEINT",
    "UTINYINT",
    "USMALLINT",
    "UINTEGER",
    "UBIGINT",
    "FLOAT",
    "DOUBLE",
    "DECIMAL",
    "REAL",
)
_TEMPORAL_PREFIXES = ("DATE", "TIME", "TIMESTAMP", "INTERVAL")
_BOOLEAN_PREFIXES = ("BOOLEAN", "BOOL")


def classify_type(duckdb_type: str) -> str:
    """Bucket a DuckDB type into numeric | temporal | boolean | categorical.

    Tools and charts need to route on a coarse kind, not on 30 concrete SQL types.
    A line chart needs a temporal or numeric x-axis; a mean needs a numeric column;
    a group-by wants a categorical one. Deciding that here means every consumer asks
    the same question the same way.
    """
    upper = duckdb_type.upper()
    if upper.startswith(_BOOLEAN_PREFIXES):
        return "boolean"
    if upper.startswith(_TEMPORAL_PREFIXES):
        return "temporal"
    if upper.startswith(_NUMERIC_PREFIXES):
        return "numeric"
    return "categorical"


@dataclass(frozen=True)
class ColumnStats:
    name: str
    position: int
    duckdb_type: str
    semantic_type: str
    null_count: int
    null_fraction: float
    distinct_count: int | None
    min_value: str | None
    max_value: str | None
    mean_value: float | None
    stddev_value: float | None
    q25_value: float | None
    q50_value: float | None
    q75_value: float | None
    is_constant: bool
    is_high_cardinality: bool


@dataclass(frozen=True)
class DatasetProfile:
    row_count: int
    column_count: int
    duplicate_row_count: int
    columns: list[ColumnStats] = field(default_factory=list)

    @property
    def total_null_fraction(self) -> float:
        """Share of all cells that are null. A headline data-health number."""
        cells = self.row_count * self.column_count
        if cells == 0:
            return 0.0
        return sum(c.null_count for c in self.columns) / cells


def _to_float(value: object) -> float | None:
    """Coerce a SUMMARIZE cell to float, or None if it is not numeric.

    SUMMARIZE returns these columns as VARCHAR because one column of output has to
    hold values from every column of input. For a text column, 'max' might be the
    word "zebra".
    """
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    # NaN and infinity are not storable as Postgres double precision values and are
    # not meaningful statistics; treat them as absent.
    if result != result or result in (float("inf"), float("-inf")):
        return None
    return result


def _quote_ident(name: str) -> str:
    """Quote a column identifier for SQL.

    Column names come from an uploaded file, so they can contain spaces, quotes,
    keywords, or non-ASCII text. Doubling embedded quotes and wrapping in double
    quotes is what makes `SELECT count(*) FROM t WHERE "weird ""name" IS NULL` valid
    instead of a syntax error — or worse, an injection.
    """
    return '"' + name.replace('"', '""') + '"'


def profile_parquet(parquet_path: Path) -> DatasetProfile:
    """Compute the full profile of a stored Parquet file.

    Reads directly from Parquet without loading it into memory: DuckDB streams and
    uses per-column-chunk statistics, so this stays usable on multi-million-row files.
    """
    source = f"read_parquet('{parquet_path.as_posix()}')"
    con = duckdb.connect(":memory:")
    try:
        row_count = int(con.execute(f"SELECT count(*) FROM {source}").fetchone()[0])

        con.execute(f"SELECT * FROM {source} LIMIT 0")
        column_names = [d[0] for d in con.description]
        column_types = [str(d[1]) for d in con.description]
        column_count = len(column_names)

        duplicate_row_count = _count_duplicate_rows(con, source, row_count)
        summary = _summarize(con, source)
        null_counts = _exact_null_counts(con, source, column_names)
        distinct_counts = _exact_distinct_counts(con, source, column_names)

        columns = [
            _build_column_stats(
                name=name,
                position=position,
                duckdb_type=dtype,
                summary_row=summary.get(name, {}),
                null_count=null_counts[name],
                distinct_count=distinct_counts[name],
                row_count=row_count,
            )
            for position, (name, dtype) in enumerate(zip(column_names, column_types, strict=True))
        ]

        return DatasetProfile(
            row_count=row_count,
            column_count=column_count,
            duplicate_row_count=duplicate_row_count,
            columns=columns,
        )
    finally:
        con.close()


def _count_duplicate_rows(con: duckdb.DuckDBPyConnection, source: str, row_count: int) -> int:
    """Rows that are exact copies of another row.

    Computed as total minus distinct, so N identical rows count as N-1 duplicates —
    the number you would remove to deduplicate, which is the number a user wants.
    """
    if row_count == 0:
        return 0
    distinct = int(
        con.execute(f"SELECT count(*) FROM (SELECT DISTINCT * FROM {source})").fetchone()[0]
    )
    return row_count - distinct


def _summarize(con: duckdb.DuckDBPyConnection, source: str) -> dict[str, dict]:
    """Run SUMMARIZE and index the result by column name."""
    result = con.execute(f"SUMMARIZE SELECT * FROM {source}")
    fields = [d[0] for d in result.description]
    rows = result.fetchall()
    out: dict[str, dict] = {}
    for row in rows:
        record = dict(zip(fields, row, strict=True))
        out[record["column_name"]] = record
    return out


def _exact_null_counts(
    con: duckdb.DuckDBPyConnection, source: str, column_names: list[str]
) -> dict[str, int]:
    """Exact null count per column, in a single scan.

    SUMMARIZE reports a null *percentage*, already rounded. Rounding is fine for
    display and wrong for arithmetic: 0.4% of 2.4M rows rounds to a number that is off
    by thousands. Since these counts end up in the profile the agent reasons about,
    they are computed exactly.
    """
    if not column_names:
        return {}
    projections = ", ".join(
        f"count(*) FILTER (WHERE {_quote_ident(name)} IS NULL)" for name in column_names
    )
    row = con.execute(f"SELECT {projections} FROM {source}").fetchone()
    return dict(zip(column_names, (int(v) for v in row), strict=True))


def _exact_distinct_counts(
    con: duckdb.DuckDBPyConnection, source: str, column_names: list[str]
) -> dict[str, int]:
    """Exact distinct value count per column, in a single scan.

    NOT SUMMARIZE's `approx_unique`, which is a HyperLogLog estimate. Measured on a
    30-row fixture with 30 genuinely distinct values, `approx_unique` returned 27 —
    a 10% error, enough to flip a cardinality threshold and to mislead anyone reading
    the profile.

    This is the same argument that makes `_exact_null_counts` exist: these numbers are
    shown to users and, in M5, reasoned over by the agent. An estimate presented as a
    count is a small lie that propagates.

    Cost: this is paid once per ingest, never per query. `count(DISTINCT ...)` follows
    SQL semantics and excludes NULLs, so this is the number of distinct *non-null*
    values.
    """
    if not column_names:
        return {}
    projections = ", ".join(f"count(DISTINCT {_quote_ident(n)})" for n in column_names)
    row = con.execute(f"SELECT {projections} FROM {source}").fetchone()
    return dict(zip(column_names, (int(v) for v in row), strict=True))


def _build_column_stats(
    *,
    name: str,
    position: int,
    duckdb_type: str,
    summary_row: dict,
    null_count: int,
    distinct_count: int | None,
    row_count: int,
) -> ColumnStats:
    semantic_type = classify_type(duckdb_type)

    # A column with one distinct value and no nulls carries no information: it cannot
    # explain variation in anything, and grouping by it yields a single bucket.
    is_constant = bool(row_count > 0 and distinct_count == 1 and null_count == 0)

    is_high_cardinality = bool(
        semantic_type == "categorical"
        and row_count >= HIGH_CARDINALITY_MIN_ROWS
        and distinct_count is not None
        and distinct_count > row_count * HIGH_CARDINALITY_FRACTION
    )

    return ColumnStats(
        name=name,
        position=position,
        duckdb_type=duckdb_type,
        semantic_type=semantic_type,
        null_count=null_count,
        null_fraction=(null_count / row_count) if row_count else 0.0,
        distinct_count=distinct_count,
        min_value=_as_text(summary_row.get("min")),
        max_value=_as_text(summary_row.get("max")),
        mean_value=_to_float(summary_row.get("avg")),
        stddev_value=_to_float(summary_row.get("std")),
        q25_value=_to_float(summary_row.get("q25")),
        q50_value=_to_float(summary_row.get("q50")),
        q75_value=_to_float(summary_row.get("q75")),
        is_constant=is_constant,
        is_high_cardinality=is_high_cardinality,
    )


def _as_text(value: object) -> str | None:
    """Store min/max as text.

    They can be numbers, dates or strings depending on the column, and one database
    column has to hold all of them. Text preserves the value for display; anything
    needing the typed value re-queries the data.
    """
    return None if value is None else str(value)[:500]
