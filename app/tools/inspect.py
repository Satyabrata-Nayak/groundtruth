"""Tools that describe the data before anything is computed from it.

WHY THESE EXIST SEPARATELY FROM `execute_sql`
---------------------------------------------
A model could discover the schema with `SELECT * FROM dataset LIMIT 5`. That is worse
in three ways: it returns raw rows the model has to eyeball and generalise from, it
says nothing about nulls or cardinality, and it spends a whole turn to learn something
we already computed at ingest.

`inspect_schema` is also the tool that makes every later query *possible* -- a model
cannot write correct SQL against columns it has not seen. In M5 the schema is likely
to be injected into the first prompt automatically, but the tool stays: on a 40-column
dataset the agent needs to be able to look again, and a question about a specific
column should not require re-reading all forty.

DEGRADED MODE IS EXPLICIT
-------------------------
Column statistics live in Postgres. If it is unreachable, these return the schema
anyway with `profile_available: false` rather than failing. The model can still do
useful work with names and types; it just must not claim anything about nulls. That
distinction is reported rather than hidden, because a silent fallback that omits null
counts would lead the model to state averages as if nothing were missing.
"""

from __future__ import annotations

from typing import Any

from app.data.profile import ColumnStats, DatasetProfile
from app.data.sandbox import TABLE_NAME
from app.data.service import get_stored_profile
from app.db.session import session_scope
from app.tools._common import dataset_columns, jsonable, resolve_column, run_sql
from app.tools.base import Tool, ToolContext


def _load_profile(context: ToolContext) -> DatasetProfile | None:
    """Stored profile for this version, or None if the metadata store is unreachable."""
    try:
        with session_scope() as session:
            return get_stored_profile(session, context.dataset_id, context.version)
    except Exception:  # noqa: BLE001 - availability problem, not a tool failure
        return None


def _stats_by_name(profile: DatasetProfile | None) -> dict[str, ColumnStats]:
    return {} if profile is None else {c.name: c for c in profile.columns}


class InspectSchemaTool(Tool):
    name = "inspect_schema"
    description = (
        "List the columns of the dataset with their data types and data-quality "
        "statistics (null counts, distinct counts, constant and high-cardinality "
        "flags). Call this before writing any SQL so that column names and types are "
        "known rather than guessed."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "include_statistics": {
                "type": "boolean",
                "description": (
                    "Include null counts, distinct counts and quality flags. "
                    "Set false for a compact list of names and types only."
                ),
                "default": True,
            }
        },
        "required": [],
    }

    def execute(
        self, context: ToolContext, *, include_statistics: bool = True
    ) -> tuple[dict[str, Any], str]:
        columns = dataset_columns(context)
        profile = _load_profile(context) if include_statistics else None
        stats = _stats_by_name(profile)

        described: list[dict[str, Any]] = []
        for column in columns:
            entry: dict[str, Any] = {
                "name": column.name,
                "type": column.duckdb_type,
                "kind": column.semantic_type,
            }
            stat = stats.get(column.name)
            if stat is not None:
                entry["null_count"] = stat.null_count
                entry["null_fraction"] = round(stat.null_fraction, 4)
                entry["distinct_count"] = stat.distinct_count
                if stat.is_constant:
                    entry["warning"] = "constant: every row has the same value"
                elif stat.is_high_cardinality:
                    entry["warning"] = (
                        "high cardinality: behaves like an identifier, "
                        "grouping by it is unlikely to be meaningful"
                    )
            described.append(entry)

        data: dict[str, Any] = {
            "columns": described,
            "column_count": len(columns),
            "profile_available": profile is not None,
        }

        if profile is not None:
            data["row_count"] = profile.row_count
            data["duplicate_row_count"] = profile.duplicate_row_count
        else:
            # Without the stored profile there is no cached row count, and one cheap
            # aggregate beats leaving the model to guess how big the dataset is.
            counted = run_sql(context, f'SELECT count(*) FROM "{TABLE_NAME}"', max_rows=1)
            data["row_count"] = int(counted.rows[0][0]) if counted.rows else 0
            data["note"] = (
                "column statistics are unavailable; null and distinct counts are not "
                "included in this result"
            )

        return data, f"{data['row_count']} rows, {len(columns)} columns"


class ProfileColumnTool(Tool):
    name = "profile_column"
    description = (
        "Get detailed statistics for one column: type, null count, distinct count, "
        "min/max, and either the most frequent values (for text columns) or the mean, "
        "standard deviation and quartiles (for numeric columns). Use this to "
        "understand a single column in depth before analysing it."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "column": {
                "type": "string",
                "description": (
                    "Name of the column to profile, exactly as it appears in the schema."
                ),
            },
            "top_values": {
                "type": "integer",
                "description": "How many of the most frequent values to return (text columns).",
                "default": 10,
                "minimum": 1,
                "maximum": 50,
            },
        },
        "required": ["column"],
    }

    def execute(
        self, context: ToolContext, *, column: str, top_values: int = 10
    ) -> tuple[dict[str, Any], str]:
        ref = resolve_column(context, column, argument="column")
        profile = _load_profile(context)
        stat = _stats_by_name(profile).get(ref.name)

        data: dict[str, Any] = {
            "column": ref.name,
            "type": ref.duckdb_type,
            "kind": ref.semantic_type,
            "profile_available": stat is not None,
        }

        if stat is not None:
            data.update(
                {
                    "null_count": stat.null_count,
                    "null_fraction": round(stat.null_fraction, 4),
                    "distinct_count": stat.distinct_count,
                    "min": stat.min_value,
                    "max": stat.max_value,
                    "is_constant": stat.is_constant,
                    "is_high_cardinality": stat.is_high_cardinality,
                }
            )
            if ref.semantic_type == "numeric":
                data.update(
                    {
                        "mean": stat.mean_value,
                        "stddev": stat.stddev_value,
                        "q25": stat.q25_value,
                        "median": stat.q50_value,
                        "q75": stat.q75_value,
                    }
                )

        # Value frequency is what SUMMARIZE does not give and what a model most needs
        # for a categorical column: which values exist, and how big each one is.
        if ref.semantic_type in ("categorical", "boolean"):
            result = run_sql(
                context,
                f"SELECT {ref.quoted} AS value, count(*) AS n "
                f'FROM "{TABLE_NAME}" GROUP BY 1 ORDER BY n DESC, 1 LIMIT {int(top_values)}',
                max_rows=top_values,
            )
            data["top_values"] = [
                {"value": jsonable(row[0]), "count": int(row[1])} for row in result.rows
            ]

        summary = f"{ref.name} ({ref.duckdb_type})"
        if stat is not None:
            summary += f": {stat.distinct_count} distinct, {stat.null_count} null"
        return data, summary
