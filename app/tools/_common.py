"""Shared plumbing for tools: column resolution, SQL identifier quoting, JSON coercion.

THE MOST IMPORTANT FUNCTION HERE IS `resolve_column`
----------------------------------------------------
Four of the six tools accept a *column name from the model* and put it into SQL.
`execute_sql` is safe because the whole statement goes through the sandbox's AST
allowlist, but `compare_groups(group_column="...")` builds SQL by string assembly, and
string assembly plus untrusted input is the shape of every SQL injection ever written.

The defence is not quoting. It is that the model's string is never used at all:

    model says  "Reveune"
                    |
                    v  resolve_column  -- looked up in the REAL schema
                    |
    SQL gets    "revenue"       <- the canonical name from DuckDB, not the model's text

A name that does not match a real column never reaches SQL; it becomes a `ToolError`
naming the columns that do exist. Quoting is still applied on the way out, because a
genuine column name can legitimately contain a space or a quote character -- but by
then the value is ours, not the model's.

Case-insensitive matching is deliberate. Small models get casing wrong constantly
("Revenue" for `revenue`), and failing that call teaches nothing except that the model
cannot type. Since the *canonical* name is what proceeds, accepting a case variant
costs no safety.
"""

from __future__ import annotations

import datetime as dt
import decimal
import difflib
import uuid
from dataclasses import dataclass
from typing import Any

from app.data import sandbox
from app.data.profile import classify_type
from app.tools.base import ToolContext, ToolError


@dataclass(frozen=True)
class ColumnRef:
    """A column that is known to exist, with its canonical spelling."""

    name: str
    duckdb_type: str
    semantic_type: str  # numeric | temporal | boolean | categorical

    @property
    def quoted(self) -> str:
        return quote_ident(self.name)


def quote_ident(name: str) -> str:
    """Quote an identifier for DuckDB.

    Doubling embedded double-quotes is the SQL-standard escape. Only ever called with
    a name that came out of the live schema.
    """
    return '"' + name.replace('"', '""') + '"'


def dataset_columns(context: ToolContext) -> list[ColumnRef]:
    """The live schema, as the query layer sees it.

    Read from the sandbox view rather than from stored metadata in Postgres, so a tool
    can never build SQL against columns that a stale profile row claims exist.
    """
    return [
        ColumnRef(name=name, duckdb_type=duckdb_type, semantic_type=classify_type(duckdb_type))
        for name, duckdb_type in sandbox.get_schema(context.dataset_id, context.version)
    ]


def resolve_column(
    context: ToolContext,
    supplied: str,
    *,
    argument: str,
    require: str | tuple[str, ...] | None = None,
    columns: list[ColumnRef] | None = None,
) -> ColumnRef:
    """Turn a model-supplied column name into a verified ColumnRef, or raise.

    `require` constrains the semantic type -- asking for the mean of a text column is
    a mistake worth catching before DuckDB produces a confusing cast error.

    Pass `columns` when resolving several names in one call, to avoid re-reading the
    schema per name.
    """
    available = dataset_columns(context) if columns is None else columns
    by_exact = {c.name: c for c in available}

    match = by_exact.get(supplied)
    if match is None:
        lowered = {c.name.lower(): c for c in available}
        match = lowered.get(supplied.lower().strip())

    if match is None:
        near = difflib.get_close_matches(supplied, list(by_exact), n=3, cutoff=0.6)
        hint = f" Did you mean: {', '.join(near)}?" if near else ""
        raise ToolError(
            f"argument '{argument}': column {supplied!r} does not exist.{hint} "
            f"Available columns: {', '.join(by_exact)}"
        )

    if require is not None:
        wanted = (require,) if isinstance(require, str) else require
        if match.semantic_type not in wanted:
            candidates = [c.name for c in available if c.semantic_type in wanted]
            raise ToolError(
                f"argument '{argument}': column '{match.name}' is "
                f"{match.semantic_type} ({match.duckdb_type}), but this tool needs a "
                f"{' or '.join(wanted)} column. "
                f"Suitable columns: {', '.join(candidates) or 'none in this dataset'}"
            )

    return match


def jsonable(value: Any) -> Any:
    """Coerce a DuckDB value into something JSON can carry.

    DuckDB hands back `Decimal`, `date`, `datetime`, `time` and `UUID` objects. These
    are destined for a JSON tool payload, so they are converted once here rather than
    crashing `json.dumps` deep inside the agent loop. Decimals become floats because a
    model reasons about numbers, not about exact decimal representation -- and the
    authoritative value stays in the database either way.
    """
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, decimal.Decimal):
        return float(value)
    if isinstance(value, (dt.datetime, dt.date, dt.time)):
        return value.isoformat()
    if isinstance(value, uuid.UUID):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    return str(value)


def run_sql(context: ToolContext, sql: str, *, max_rows: int) -> sandbox.QueryResult:
    """Run tool-generated SQL through the same sandbox model-written SQL goes through.

    Tool SQL is assembled from verified column names and could reasonably skip
    validation. It does not, for two reasons: one code path is one set of bugs, and a
    tool whose SQL somehow became malformed should fail the same clean way a model's
    would.
    """
    try:
        return sandbox.execute_sql(
            context.dataset_id, context.version, sql, max_rows=max_rows
        )
    except sandbox.SqlValidationError as exc:
        raise ToolError(f"generated SQL was rejected: {exc}") from exc
    except sandbox.SqlExecutionError as exc:
        raise ToolError(f"query failed: {exc}") from exc
    except FileNotFoundError as exc:
        raise ToolError(str(exc)) from exc


def rows_as_lists(result: sandbox.QueryResult) -> list[list[Any]]:
    """Rows as JSON-safe lists.

    Lists rather than dicts: the column names are already carried once alongside, and
    repeating them per row triples the token cost of a result for no added meaning.
    """
    return [[jsonable(v) for v in row] for row in result.rows]
