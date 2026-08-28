"""Statistical tools: comparing groups, and measuring relationships between columns.

WHY THESE EXIST WHEN `execute_sql` COULD DO BOTH
------------------------------------------------
Both of these are expressible in SQL, so neither adds raw capability. They add three
things SQL does not:

1. TYPE ENFORCEMENT BEFORE EXECUTION. `corr(region, revenue)` on a text column is a
   cast error from DuckDB; here it is "column 'region' is categorical, this tool needs
   a numeric column. Suitable columns: revenue, cost, units". One is a dead end; the
   other is a repair instruction.

2. THE SECOND NUMBER THE MODEL WOULD FORGET. `compare_groups` returns the row count
   and the share of total alongside each group's metric, because "region North has the
   highest average order value" is misleading when North has four orders. A model
   writing its own SQL asks for the average and stops.

3. A GUARD AGAINST MEANINGLESS GROUPINGS. Grouping by a high-cardinality column
   produces one row per record. That is not an aggregation, it is the raw table with
   extra steps, and it is refused.

PEARSON AND SPEARMAN, NOT JUST PEARSON
--------------------------------------
Pearson measures *linear* association. A relationship that is strong but curved --
diminishing returns on ad spend, a saturating conversion rate -- can show a weak
Pearson value and be missed entirely. Spearman is Pearson computed on the ranks, so it
detects any monotone relationship, straight or not. Computing both costs one extra
window function, and a large gap between them is itself the finding: it says
"related, but not in a straight line".
"""

from __future__ import annotations

import math
from typing import Any

from app.data.sandbox import TABLE_NAME
from app.tools._common import ColumnRef, dataset_columns, jsonable, resolve_column, run_sql
from app.tools.base import Tool, ToolContext, ToolError

# SQL aggregate for each name the model may ask for. An allowlist, not a format
# string: the value is chosen by us from a fixed set, so the model's string never
# becomes SQL even though it names a SQL function.
_AGGREGATIONS: dict[str, str] = {
    "sum": "sum({col})",
    "avg": "avg({col})",
    "mean": "avg({col})",
    "median": "median({col})",
    "min": "min({col})",
    "max": "max({col})",
    "count": "count({col})",
    "count_distinct": "count(DISTINCT {col})",
}

# Grouping by a column with more distinct values than this returns more groups than
# anyone reads, whatever the table size.
_MAX_GROUPS = 1000

# A column whose values are nearly all distinct is an identifier, not a category.
# Same thresholds as `app.data.profile.HIGH_CARDINALITY_*`, deliberately: two places
# calling the same thing "identifier-like" by different rules is a bug waiting to
# confuse someone.
_IDENTIFIER_FRACTION = 0.9
_IDENTIFIER_MIN_ROWS = 20


class CompareGroupsTool(Tool):
    name = "compare_groups"
    description = (
        "Aggregate a numeric metric across the categories of another column and rank "
        "the results. Answers questions of the form 'which X has the highest Y' or "
        "'how does Y differ between groups of X'. Returns each group's value, its row "
        "count and its share of the total, so that small groups are visible."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "group_column": {
                "type": "string",
                "description": "The column defining the groups, e.g. a region or category.",
            },
            "metric_column": {
                "type": "string",
                "description": "The numeric column to aggregate within each group.",
            },
            "aggregation": {
                "type": "string",
                "description": "How to aggregate the metric within each group.",
                "enum": sorted(_AGGREGATIONS),
                "default": "sum",
            },
            "order": {
                "type": "string",
                "description": "Sort the groups by the aggregated value.",
                "enum": ["desc", "asc"],
                "default": "desc",
            },
            "limit": {
                "type": "integer",
                "description": "How many groups to return.",
                "default": 20,
                "minimum": 1,
                "maximum": 100,
            },
        },
        "required": ["group_column", "metric_column"],
    }

    def execute(
        self,
        context: ToolContext,
        *,
        group_column: str,
        metric_column: str,
        aggregation: str = "sum",
        order: str = "desc",
        limit: int = 20,
    ) -> tuple[dict[str, Any], str]:
        columns = dataset_columns(context)
        group = resolve_column(context, group_column, argument="group_column", columns=columns)

        # `count` is the one aggregation that is meaningful on any type; everything
        # else needs a number.
        needs_numeric = aggregation not in ("count", "count_distinct")
        metric = resolve_column(
            context,
            metric_column,
            argument="metric_column",
            require="numeric" if needs_numeric else None,
            columns=columns,
        )

        _reject_unusable_grouping(context, group)

        aggregate = _AGGREGATIONS[aggregation].format(col=metric.quoted)
        direction = "DESC" if order == "desc" else "ASC"
        sql = (
            f"SELECT {group.quoted} AS group_value, "
            f"{aggregate} AS metric_value, "
            f"count(*) AS row_count "
            f'FROM "{TABLE_NAME}" '
            f"GROUP BY {group.quoted} "
            f"ORDER BY metric_value {direction} NULLS LAST, group_value "
            f"LIMIT {int(limit)}"
        )
        result = run_sql(context, sql, max_rows=limit)

        values = [row[1] for row in result.rows if isinstance(row[1], (int, float))]
        total = float(sum(values)) if values else 0.0
        # A share of total is only meaningful for an additive aggregate. The average of
        # averages is not the overall average, so shares are withheld rather than
        # offered in a form a model would reasonably misread.
        additive = aggregation in ("sum", "count", "count_distinct")

        groups: list[dict[str, Any]] = []
        for row in result.rows:
            entry: dict[str, Any] = {
                "group": jsonable(row[0]),
                "value": jsonable(row[1]),
                "row_count": int(row[2]),
            }
            if additive and total:
                entry["share_of_total"] = round(float(row[1]) / total, 4)
            groups.append(entry)

        data: dict[str, Any] = {
            "group_column": group.name,
            "metric_column": metric.name,
            "aggregation": aggregation,
            "groups": groups,
            "group_count_returned": len(groups),
            "truncated": result.truncated,
        }
        if additive:
            data["total_across_returned_groups"] = total

        if not groups:
            return data, "no groups found"

        best = groups[0]
        return data, (
            f"{len(groups)} group(s) by {group.name}; top: {best['group']} = {best['value']}"
        )


def _reject_unusable_grouping(context: ToolContext, group: ColumnRef) -> None:
    """Refuse a grouping that would return roughly one row per record.

    TWO RULES, BECAUSE ONE ABSOLUTE CAP IS THE WRONG SHAPE
    ------------------------------------------------------
    The first version of this used only an absolute limit of 1,000 distinct values,
    and duly accepted `compare_groups(group_column="order_id")` on a 400-row table --
    400 groups of one row each, which is the raw table wearing a hat. Uniqueness is
    relative: 400 distinct values is a fine grouping in a million rows and meaningless
    in four hundred.

    So the identifier test is the same fraction rule `app.data.profile` uses for its
    high-cardinality flag, and the absolute cap stays alongside it to bound the result
    size for a genuinely categorical column with very many values.

    Both are checked against a live count rather than the stored flag: the flag is a
    heuristic recorded at ingest, and these are limits.
    """
    counted = run_sql(
        context,
        f'SELECT count(DISTINCT {group.quoted}), count(*) FROM "{TABLE_NAME}"',
        max_rows=1,
    )
    distinct, rows = (int(counted.rows[0][0]), int(counted.rows[0][1])) if counted.rows else (0, 0)

    if rows >= _IDENTIFIER_MIN_ROWS and distinct >= rows * _IDENTIFIER_FRACTION:
        raise ToolError(
            f"column '{group.name}' has {distinct} distinct values across {rows} rows, "
            f"so grouping by it produces about one group per row and aggregates "
            f"nothing. It behaves like an identifier. Group by a column with real "
            f"categories, or use execute_sql if you meant to inspect individual rows."
        )

    if distinct > _MAX_GROUPS:
        raise ToolError(
            f"column '{group.name}' has {distinct} distinct values, which is too many "
            f"to compare as groups (limit {_MAX_GROUPS}). Group by a coarser column, "
            f"or use execute_sql with your own filter."
        )


class CorrelationTool(Tool):
    name = "correlation"
    description = (
        "Measure the strength and direction of the relationship between two numeric "
        "columns. Returns the Pearson coefficient (straight-line relationships) and "
        "the Spearman coefficient (any consistently increasing or decreasing "
        "relationship, including curved ones), plus the number of rows used."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "column_a": {"type": "string", "description": "First numeric column."},
            "column_b": {"type": "string", "description": "Second numeric column."},
        },
        "required": ["column_a", "column_b"],
    }

    def execute(
        self, context: ToolContext, *, column_a: str, column_b: str
    ) -> tuple[dict[str, Any], str]:
        columns = dataset_columns(context)
        first = resolve_column(
            context, column_a, argument="column_a", require="numeric", columns=columns
        )
        second = resolve_column(
            context, column_b, argument="column_b", require="numeric", columns=columns
        )

        if first.name == second.name:
            raise ToolError(
                f"column_a and column_b are both '{first.name}'. A column correlates "
                f"perfectly with itself; pass two different columns."
            )

        # Rows where either value is null are excluded from both coefficients, so the
        # two are computed over the same population and `n` describes both.
        both_present = (
            f'FROM "{TABLE_NAME}" WHERE {first.quoted} IS NOT NULL AND {second.quoted} IS NOT NULL'
        )
        sql = (
            f"WITH paired AS (SELECT {first.quoted} AS a, {second.quoted} AS b {both_present}), "
            f"ranked AS (SELECT rank() OVER (ORDER BY a) AS ra, "
            f"rank() OVER (ORDER BY b) AS rb FROM paired) "
            f"SELECT (SELECT count(*) FROM paired) AS n, "
            f"(SELECT corr(a, b) FROM paired) AS pearson, "
            f"(SELECT corr(ra, rb) FROM ranked) AS spearman"
        )
        result = run_sql(context, sql, max_rows=1)
        if not result.rows:
            raise ToolError("correlation returned no result")

        n_raw, pearson_raw, spearman_raw = result.rows[0]
        n = int(n_raw)
        if n < 3:
            raise ToolError(
                f"only {n} row(s) have a value in both '{first.name}' and "
                f"'{second.name}'. A correlation needs at least 3."
            )

        pearson = _finite(pearson_raw)
        spearman = _finite(spearman_raw)

        data: dict[str, Any] = {
            "column_a": first.name,
            "column_b": second.name,
            "rows_used": n,
            "pearson": None if pearson is None else round(pearson, 4),
            "spearman": None if spearman is None else round(spearman, 4),
            "strength": _strength(pearson),
            "direction": _direction(pearson),
        }

        if pearson is None:
            data["note"] = (
                "the coefficient is undefined, which happens when one of the columns "
                "has no variation in the rows where both are present"
            )
        elif spearman is not None and abs(spearman) - abs(pearson) > 0.2:
            data["note"] = (
                "Spearman is much stronger than Pearson: the relationship is "
                "consistent but not a straight line"
            )

        if pearson is None:
            return data, f"{first.name} vs {second.name}: undefined (n={n})"
        return data, (
            f"{first.name} vs {second.name}: pearson={pearson:.3f} "
            f"({data['strength']} {data['direction']}), n={n}"
        )


def _finite(value: Any) -> float | None:
    """None for NULL, NaN and infinity -- all of which mean 'no coefficient here'."""
    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _strength(value: float | None) -> str:
    """Plain-English magnitude, so the model does not have to invent a threshold.

    The cut points are the conventional social-science ones. They are a convention,
    not a fact about the world, which is why the numeric coefficient is returned
    alongside and is what any claim should ultimately cite.
    """
    if value is None:
        return "undefined"
    magnitude = abs(value)
    if magnitude >= 0.7:
        return "strong"
    if magnitude >= 0.4:
        return "moderate"
    if magnitude >= 0.2:
        return "weak"
    return "negligible"


def _direction(value: float | None) -> str:
    if value is None:
        return "none"
    if value > 0:
        return "positive"
    if value < 0:
        return "negative"
    return "flat"
