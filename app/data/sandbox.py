"""Execute analytical SQL against one dataset, treating the SQL as hostile input.

In M5 this SQL is written by a language model. Long before that, it should be treated
as untrusted: a model can be steered by text inside the dataset itself (prompt
injection), and "the model would never write that" is not a security control.

FOUR LAYERS, BECAUSE EACH COVERS WHAT THE OTHERS MISS
-----------------------------------------------------

    L1  parse    sqlglot: exactly ONE statement, root node must be SELECT or WITH.
                 An ALLOWLIST. Keyword blocklists lose to comments, nesting and
                 string literals:
                     SELECT 1; /*x*/ DROP TABLE t
                     WITH a AS (SELECT 1) SELECT * FROM read_csv('~/.ssh/id_rsa')

    L2  confine  A fresh DuckDB connection per query, locked down so the process
                 cannot reach the filesystem or network at all. THIS is the layer that
                 actually provides safety; L1 and L3 make it explainable and give
                 clean error messages.

    L3  scope    Only this dataset's Parquet file is registered, as a view. The caller
                 passes a dataset_id; no path ever appears in SQL.

    L4  bound    Row cap, byte cap, wall-clock timeout — so a legal query cannot
                 exhaust memory or hang a worker.

THE LOCKDOWN RECIPE, AND WHY THE ORDER MATTERS
----------------------------------------------
Verified empirically against DuckDB 1.5.5 (see docs/decisions.md D-011):

    SET memory_limit / threads        <- must be FIRST; frozen by the lock below
    SET allowed_paths = [our parquet] <- the ONE file that stays readable
    SET enable_external_access=false  <- everything else off
    SET lock_configuration=true       <- a query cannot undo any of the above

The naive approach — `enable_external_access=false` alone — also blocks reading our
own Parquet file, which is why `allowed_paths` is needed: DuckDB documents it as files
that are "ALWAYS allowed to be queried, even when external access is disabled".
"""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb
import sqlglot
import sqlglot.expressions as sqlexp

from app.config import get_settings
from app.data import storage

# The table name SQL sees. Fixed rather than derived from the filename so that queries
# are predictable, no filename can leak into SQL, and the schema shown to the model in
# M5 always matches what the query must say.
TABLE_NAME = "dataset"

# Only these can be the ROOT of a statement. Anything else — INSERT, UPDATE, DELETE,
# DROP, ATTACH, COPY, PRAGMA, SET, CALL, EXPORT — is rejected before execution.
_ALLOWED_ROOTS = (sqlexp.Select, sqlexp.Union, sqlexp.Except, sqlexp.Intersect)


class SqlValidationError(ValueError):
    """The SQL was rejected before execution. Safe to show a user or feed back to a model."""


class SqlExecutionError(RuntimeError):
    """The SQL parsed and was permitted, but failed or exceeded a limit while running."""


@dataclass(frozen=True)
class QueryResult:
    columns: list[str]
    rows: list[tuple[Any, ...]]
    row_count: int
    truncated: bool
    execution_ms: float

    def as_dicts(self) -> list[dict[str, Any]]:
        return [dict(zip(self.columns, row, strict=True)) for row in self.rows]


# --------------------------------------------------------------------------------
# L1 — parse and allowlist
# --------------------------------------------------------------------------------


def validate_sql(sql: str) -> sqlexp.Expression:
    """Parse SQL and confirm it is a single read-only statement. Returns the AST.

    Rejects on structure, not on spelling. `DROP` hidden in a comment, a second
    statement after a semicolon, or a CTE wrapping a write are all caught because the
    parse tree says what the statement *is*, regardless of how it is written.
    """
    if not sql or not sql.strip():
        raise SqlValidationError("query is empty")

    try:
        statements = sqlglot.parse(sql, read="duckdb")
    except Exception as exc:  # sqlglot raises several types
        raise SqlValidationError(f"could not parse SQL: {str(exc).splitlines()[0][:200]}") from exc

    # `parse` returns None entries for empty statements, e.g. a trailing semicolon.
    statements = [s for s in statements if s is not None]

    if not statements:
        raise SqlValidationError("no SQL statement found")
    if len(statements) > 1:
        raise SqlValidationError(
            f"expected exactly one statement, found {len(statements)}. "
            "Multiple statements are not permitted."
        )

    root = statements[0]

    # A WITH-prefixed statement parses as the inner expression carrying a `with` arg,
    # so checking the root type also covers CTEs. A CTE wrapping an INSERT therefore
    # fails here, because the root is an Insert, not a Select.
    if not isinstance(root, _ALLOWED_ROOTS):
        raise SqlValidationError(
            f"only SELECT queries are permitted, got {type(root).__name__.upper()}"
        )

    _reject_forbidden_nodes(root)
    return root


# Constructs that can appear nested inside an otherwise-valid SELECT.
_FORBIDDEN_NODE_TYPES: tuple[type, ...] = tuple(
    node
    for node in (
        getattr(sqlexp, name, None)
        for name in (
            "Insert",
            "Update",
            "Delete",
            "Drop",
            "Create",
            "Alter",
            "Attach",
            "Detach",
            "Copy",
            "Command",
            "Pragma",
            "Set",
            "Grant",
            "Merge",
            "Export",
            "Load",
            "Install",
        )
    )
    if node is not None
)

# Table functions that read from outside the registered dataset. L2 blocks these at the
# engine level; naming them here produces a clear message instead of a permission error,
# which matters when the message is fed back to a model for repair.
_FORBIDDEN_FUNCTIONS = {
    "read_csv",
    "read_csv_auto",
    "read_parquet",
    "read_json",
    "read_json_auto",
    "read_text",
    "read_blob",
    "read_ndjson",
    "glob",
    "parquet_scan",
    "csv_scan",
    "delta_scan",
    "iceberg_scan",
    "postgres_scan",
    "sqlite_scan",
    "mysql_scan",
}


def _reject_forbidden_nodes(root: sqlexp.Expression) -> None:
    """Walk the whole tree, including subqueries and CTE bodies."""
    for node in root.walk():
        if isinstance(node, _FORBIDDEN_NODE_TYPES):
            raise SqlValidationError(f"{type(node).__name__.upper()} is not permitted in a query")
        if isinstance(node, sqlexp.Table):
            _reject_table_function(node)
        if isinstance(node, sqlexp.Anonymous):
            name = (node.name or "").lower()
            if name in _FORBIDDEN_FUNCTIONS:
                raise SqlValidationError(
                    f"function '{name}' is not permitted; query the '{TABLE_NAME}' table instead"
                )


def _reject_table_function(table: sqlexp.Table) -> None:
    """A FROM source must be a plain named table, never a table-valued function.

    WHY THIS IS STRUCTURAL RATHER THAN A NAME LIST
    ----------------------------------------------
    The first version of this check blocklisted function names (`read_csv`,
    `read_parquet`, ...) by looking for `sqlexp.Anonymous` nodes. It silently missed
    both of those, because sqlglot gives them DEDICATED node classes — `ReadCSV` and
    `ReadParquet` — which are not `Anonymous` at all. L2 caught the query anyway, but
    L1 is what produces the clear, repairable message, and a check that misses the two
    most obvious functions is not a check.

    More importantly, a name list only ever blocks functions someone thought of.
    DuckDB ships many table functions and extensions add more. Since this sandbox
    exposes exactly ONE table, the correct rule is the structural one: every table
    reference must be a plain identifier. That rejects every table-valued function,
    including ones that do not exist yet.
    """
    inner = table.this
    if inner is None or isinstance(inner, sqlexp.Identifier):
        return
    label = getattr(inner, "name", "") or type(inner).__name__
    raise SqlValidationError(
        f"table functions are not permitted (found '{label}'); "
        f"query the '{TABLE_NAME}' table instead"
    )


# --------------------------------------------------------------------------------
# L2 + L3 — confined connection scoped to one dataset
# --------------------------------------------------------------------------------


def _open_confined_connection(parquet: Path) -> duckdb.DuckDBPyConnection:
    """A DuckDB connection that can read exactly one file and nothing else.

    Order is load-bearing: `lock_configuration` freezes every setting, so limits must
    be applied before it. Verified against DuckDB 1.5.5 — after this sequence, reading
    any other path, ATTACHing a file database, COPYing out, installing an extension,
    fetching a URL, and re-enabling external access all fail.
    """
    settings = get_settings()
    con = duckdb.connect(":memory:")

    con.execute(f"SET memory_limit='{settings.duckdb_memory_limit}'")
    con.execute(f"SET threads={int(settings.duckdb_threads)}")

    # The view must be created while the file is still reachable.
    con.execute(
        f'CREATE VIEW "{TABLE_NAME}" AS '
        f"SELECT * FROM read_parquet({storage.sql_path_literal(parquet)})"
    )

    con.execute(f"SET allowed_paths=['{parquet.as_posix()}']")
    con.execute("SET enable_external_access=false")
    con.execute("SET lock_configuration=true")
    return con


# --------------------------------------------------------------------------------
# L4 — limits
# --------------------------------------------------------------------------------


def _run_with_timeout(
    con: duckdb.DuckDBPyConnection, sql: str, timeout_s: float
) -> duckdb.DuckDBPyConnection:
    """Execute, interrupting the query if it outruns the timeout.

    DuckDB has no built-in statement timeout, so a watchdog thread calls
    `con.interrupt()`. The interrupt is cooperative — DuckDB checks for it between
    work units — which is enough for a runaway scan or cross join, the realistic
    failure modes here.
    """
    done = threading.Event()

    def watchdog() -> None:
        if not done.wait(timeout_s):
            con.interrupt()

    timer = threading.Thread(target=watchdog, daemon=True)
    timer.start()
    try:
        return con.execute(sql)
    finally:
        done.set()


def execute_sql(
    dataset_id: uuid.UUID | str,
    version: int,
    sql: str,
    *,
    max_rows: int | None = None,
    timeout_s: float | None = None,
) -> QueryResult:
    """Validate and run one read-only query against one dataset version.

    The caller supplies a dataset id, never a path (L3). The SQL is parsed and
    allowlisted (L1), executed on a confined connection (L2), and bounded (L4).
    """
    settings = get_settings()
    max_rows = settings.max_result_rows if max_rows is None else max_rows
    timeout_s = settings.query_timeout_s if timeout_s is None else timeout_s

    validate_sql(sql)  # L1
    parquet = storage.resolve_existing_parquet(dataset_id, version)  # L3

    con = _open_confined_connection(parquet)  # L2
    started = time.perf_counter()
    try:
        cursor = _run_with_timeout(con, sql, timeout_s)  # L4
        columns = [d[0] for d in cursor.description] if cursor.description else []

        # Fetch one more row than the cap so truncation is detectable rather than
        # silently indistinguishable from a result that happens to be exactly max_rows.
        fetched = cursor.fetchmany(max_rows + 1)
        truncated = len(fetched) > max_rows
        rows = fetched[:max_rows]
    except duckdb.InterruptException as exc:
        raise SqlExecutionError(f"query exceeded the {timeout_s:g}s time limit") from exc
    except duckdb.Error as exc:
        raise SqlExecutionError(str(exc).splitlines()[0][:300]) from exc
    finally:
        con.close()

    return QueryResult(
        columns=columns,
        rows=rows,
        row_count=len(rows),
        truncated=truncated,
        execution_ms=(time.perf_counter() - started) * 1000,
    )


def get_schema(dataset_id: uuid.UUID | str, version: int) -> list[tuple[str, str]]:
    """Column names and types, as the query layer sees them.

    In M5 this is what gets put in the model's prompt, so it must come from the same
    view the model will query — not from stored metadata that could drift.
    """
    parquet = storage.resolve_existing_parquet(dataset_id, version)
    con = _open_confined_connection(parquet)
    try:
        con.execute(f'SELECT * FROM "{TABLE_NAME}" LIMIT 0')
        return [(d[0], str(d[1])) for d in con.description]
    finally:
        con.close()
