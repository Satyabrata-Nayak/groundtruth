"""Build a chart *specification* -- a validated description of a chart, plus its data.

WHY A SPEC AND NOT AN IMAGE
---------------------------
This tool returns JSON describing what to draw. It does not render a PNG. Three
reasons, in order of how much they matter:

1. THE MODEL MUST NOT SEE PIXELS. A tool result goes into the context window. An image
   is either useless there or enormously expensive; a spec is a few hundred tokens and
   the model can reason about the numbers inside it.
2. RENDERING BELONGS TO THE CLIENT. The browser draws it, so the chart is interactive,
   themeable and re-renderable without re-running the analysis.
3. NO HEADLESS BROWSER. Server-side rendering means `kaleido` or a Chromium download
   -- a fragile dependency on Windows for a capability nothing here needs.

WHAT VALIDATION ACTUALLY BUYS
-----------------------------
The failure mode this prevents is not a crash, it is a *plausible nonsense chart*: a
line chart whose x-axis is an unordered category, a bar chart with 4,000 bars, a
scatter plot of text. All of those render fine and mean nothing. Each chart type
declares what its axes need, and the request is refused with the reason before any
data is fetched.

Aggregation is applied by default rather than plotting raw rows: a bar chart of
50,000 individual orders grouped by region is 50,000 overlapping bars, when what was
wanted was five.
"""

from __future__ import annotations

from typing import Any

from app.config import get_settings
from app.data.sandbox import TABLE_NAME
from app.tools._common import ColumnRef, dataset_columns, jsonable, resolve_column, run_sql
from app.tools.base import Tool, ToolContext, ToolError

# Aggregates permitted on the y-axis. Chosen by us from a fixed set; the model's
# string selects a key, it never becomes SQL itself.
_AGGREGATIONS: dict[str, str] = {
    "sum": "sum({col})",
    "avg": "avg({col})",
    "median": "median({col})",
    "min": "min({col})",
    "max": "max({col})",
    "count": "count({col})",
}


class _ChartKind:
    """What one chart type requires of its axes, and how many points it can carry.

    `max_points` differs by type because readability does. Fifty bars is already a
    crowded chart and each one needs a legible label; a line over 365 daily points is
    an ordinary time series and refusing it would make the most common chart in the
    product unbuildable. The first version of this used one cap for everything and
    rejected a perfectly good daily revenue line -- the limit was measuring the wrong
    thing, which is category labels, not data points.
    """

    def __init__(
        self,
        *,
        x_kinds: tuple[str, ...],
        needs_y: bool,
        y_kinds: tuple[str, ...] = ("numeric",),
        aggregates: bool = True,
        x_role: str = "",
        max_points: int | None = None,
    ) -> None:
        self.x_kinds = x_kinds
        self.needs_y = needs_y
        self.y_kinds = y_kinds
        self.aggregates = aggregates
        self.x_role = x_role
        self._max_points = max_points

    def max_points(self) -> int:
        return self._max_points or get_settings().max_chart_categories


_KINDS: dict[str, _ChartKind] = {
    # Categories on x, one labelled bar per category. Labels are the binding limit.
    "bar": _ChartKind(
        x_kinds=("categorical", "boolean", "temporal", "numeric"),
        needs_y=True,
        x_role="a category to put on the x-axis",
    ),
    # A line implies the x-axis has a meaningful order. Categories do not.
    # A dense line is normal -- daily points across two years is 730 and readable.
    "line": _ChartKind(
        x_kinds=("temporal", "numeric"),
        needs_y=True,
        x_role="a date/time or numeric column, because a line implies ordered x values",
        max_points=1000,
    ),
    # Raw points: aggregating would destroy the thing being shown.
    "scatter": _ChartKind(
        x_kinds=("numeric", "temporal"),
        needs_y=True,
        aggregates=False,
        x_role="a numeric column",
    ),
    # One column, bucketed by value.
    "histogram": _ChartKind(
        x_kinds=("numeric",),
        needs_y=False,
        aggregates=False,
        x_role="the numeric column whose distribution is being shown",
    ),
}


class CreateChartTool(Tool):
    name = "create_chart"
    description = (
        "Produce a chart specification and its underlying data, ready for the "
        "interface to render. Choose 'bar' to compare categories, 'line' for a trend "
        "over time, 'scatter' to show the relationship between two numeric columns, "
        "and 'histogram' for the distribution of one numeric column. Values are "
        "aggregated per x value for bar and line charts."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "chart_type": {
                "type": "string",
                "description": "The kind of chart to produce.",
                "enum": sorted(_KINDS),
            },
            "x": {
                "type": "string",
                "description": "Column for the x-axis (the column to bucket, for a histogram).",
            },
            "y": {
                "type": "string",
                "description": (
                    "Numeric column for the y-axis. Required for bar, line and "
                    "scatter; not used by histogram."
                ),
            },
            "aggregation": {
                "type": "string",
                "description": "How to combine y values sharing an x value (bar and line only).",
                "enum": sorted(_AGGREGATIONS),
                "default": "sum",
            },
            "title": {"type": "string", "description": "Human-readable chart title."},
            "bins": {
                "type": "integer",
                "description": "Number of buckets for a histogram.",
                "default": 20,
                "minimum": 2,
                "maximum": 100,
            },
        },
        "required": ["chart_type", "x"],
    }

    def execute(
        self,
        context: ToolContext,
        *,
        chart_type: str,
        x: str,
        y: str | None = None,
        aggregation: str = "sum",
        title: str | None = None,
        bins: int = 20,
    ) -> tuple[dict[str, Any], str]:
        kind = _KINDS[chart_type]
        columns = dataset_columns(context)

        x_ref = resolve_column(context, x, argument="x", require=kind.x_kinds, columns=columns)
        y_ref: ColumnRef | None = None

        if kind.needs_y:
            if y is None:
                raise ToolError(
                    f"a {chart_type} chart needs a 'y' column. 'x' should be "
                    f"{kind.x_role}, and 'y' the numeric value to plot."
                )
            y_ref = resolve_column(context, y, argument="y", require=kind.y_kinds, columns=columns)
        elif y is not None:
            raise ToolError(
                f"a {chart_type} chart does not use a 'y' column; it shows the "
                f"distribution of 'x' alone. Remove the 'y' argument."
            )

        if chart_type == "histogram":
            points, extra = _histogram(context, x_ref, bins)
        elif chart_type == "scatter":
            points, extra = _scatter(context, x_ref, y_ref)
        else:
            points, extra = _aggregated(context, x_ref, y_ref, aggregation, kind, chart_type)

        spec: dict[str, Any] = {
            "type": chart_type,
            "title": title or _default_title(chart_type, x_ref, y_ref, aggregation),
            "x": {"column": x_ref.name, "kind": x_ref.semantic_type, "label": x_ref.name},
            "data": points,
            "point_count": len(points),
        }
        if y_ref is not None:
            label = y_ref.name if chart_type == "scatter" else f"{aggregation}({y_ref.name})"
            spec["y"] = {"column": y_ref.name, "kind": "numeric", "label": label}
            if chart_type != "scatter":
                spec["aggregation"] = aggregation
        spec.update(extra)

        return {"chart": spec}, f"{chart_type} chart with {len(points)} point(s)"

    # How many data points the model is shown before the payload is summarised
    # instead. Twelve is enough to read a trend or a ranking out of a small chart,
    # and small enough that a 400-point scatter plot does not cost 3,000 tokens.
    MODEL_VISIBLE_POINTS = 12

    def model_view(self, data: dict[str, Any]) -> dict[str, Any]:
        """Describe the chart to the model instead of handing it every data point.

        The full spec still goes to the caller and on to the browser -- `data` is
        untouched. This only trims what re-enters the conversation, because the model
        already computed these numbers to request the chart and does not need them
        echoed back at token cost.
        """
        spec = dict(data["chart"])
        points = spec.get("data", [])
        if len(points) <= self.MODEL_VISIBLE_POINTS:
            return data

        spec["data"] = points[: self.MODEL_VISIBLE_POINTS]
        spec["data_truncated"] = True
        spec["note"] = (
            f"the chart contains {len(points)} points; {self.MODEL_VISIBLE_POINTS} are "
            f"shown here. The full data has been passed to the interface for rendering."
        )
        return {"chart": spec}


def _aggregated(
    context: ToolContext,
    x_ref: ColumnRef,
    y_ref: ColumnRef | None,
    aggregation: str,
    kind: _ChartKind,
    chart_type: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """One point per distinct x value, y aggregated within it."""
    assert y_ref is not None  # guaranteed by the caller's needs_y check
    limit = kind.max_points()

    distinct = run_sql(
        context, f'SELECT count(DISTINCT {x_ref.quoted}) FROM "{TABLE_NAME}"', max_rows=1
    )
    category_count = int(distinct.rows[0][0]) if distinct.rows else 0
    if category_count > limit:
        raise ToolError(
            f"column '{x_ref.name}' has {category_count} distinct values; a readable "
            f"{chart_type} chart holds at most {limit}. Aggregate to a coarser column "
            f"(date_trunc('month', ...) instead of a raw date, a category instead of a "
            f"product), or use execute_sql to select the top values first."
        )

    aggregate = _AGGREGATIONS[aggregation].format(col=y_ref.quoted)
    # Time and numbers are ordered by their value; categories by size, so the chart
    # reads as a ranking rather than an arbitrary alphabet.
    order = (
        "x_value"
        if x_ref.semantic_type in ("temporal", "numeric")
        else "y_value DESC NULLS LAST, x_value"
    )
    sql = (
        f"SELECT {x_ref.quoted} AS x_value, {aggregate} AS y_value "
        f'FROM "{TABLE_NAME}" WHERE {x_ref.quoted} IS NOT NULL '
        f"GROUP BY x_value ORDER BY {order} LIMIT {limit}"
    )
    result = run_sql(context, sql, max_rows=limit)
    points = [{"x": jsonable(row[0]), "y": jsonable(row[1])} for row in result.rows]
    return points, {}


def _scatter(
    context: ToolContext, x_ref: ColumnRef, y_ref: ColumnRef | None
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Raw pairs, sampled deterministically if there are too many to plot."""
    assert y_ref is not None
    cap = 500

    sql = (
        f"SELECT {x_ref.quoted} AS x_value, {y_ref.quoted} AS y_value "
        f'FROM "{TABLE_NAME}" '
        f"WHERE {x_ref.quoted} IS NOT NULL AND {y_ref.quoted} IS NOT NULL "
        # A stable pseudo-random order: hashing the row values gives the same sample
        # every run, so a chart is reproducible. ORDER BY random() would not be.
        f"ORDER BY hash({x_ref.quoted}, {y_ref.quoted}) LIMIT {cap}"
    )
    result = run_sql(context, sql, max_rows=cap)
    points = [{"x": jsonable(row[0]), "y": jsonable(row[1])} for row in result.rows]

    extra: dict[str, Any] = {}
    if len(points) == cap:
        extra["sampled"] = True
        extra["note"] = f"showing a deterministic sample of {cap} points"
    return points, extra


def _histogram(
    context: ToolContext, x_ref: ColumnRef, bins: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Equal-width buckets between the column's min and max."""
    bounds = run_sql(
        context,
        f"SELECT min({x_ref.quoted}), max({x_ref.quoted}), count({x_ref.quoted}) "
        f'FROM "{TABLE_NAME}"',
        max_rows=1,
    )
    low_raw, high_raw, count_raw = bounds.rows[0]
    if count_raw is None or int(count_raw) == 0:
        raise ToolError(f"column '{x_ref.name}' has no non-null values to plot")

    low, high = float(low_raw), float(high_raw)
    if low == high:
        # A constant column has no distribution. One bucket is the honest answer;
        # dividing by a zero-width range would be a crash.
        return (
            [{"bin_start": low, "bin_end": high, "count": int(count_raw)}],
            {"note": f"every value of '{x_ref.name}' is {low}"},
        )

    width = (high - low) / bins
    # `least(..., bins - 1)` folds the maximum value into the last bucket instead of
    # creating an extra bucket of exactly one element at the top edge.
    sql = (
        f"SELECT least(floor(({x_ref.quoted} - {low}) / {width}), {bins - 1}) AS bucket, "
        f"count(*) AS n "
        f'FROM "{TABLE_NAME}" WHERE {x_ref.quoted} IS NOT NULL '
        f"GROUP BY bucket ORDER BY bucket"
    )
    result = run_sql(context, sql, max_rows=bins)

    counts = {int(row[0]): int(row[1]) for row in result.rows}
    points = [
        {
            "bin_start": round(low + index * width, 6),
            "bin_end": round(low + (index + 1) * width, 6),
            "count": counts.get(index, 0),  # empty buckets must still appear
        }
        for index in range(bins)
    ]
    return points, {"bins": bins, "min": low, "max": high}


def _default_title(
    chart_type: str, x_ref: ColumnRef, y_ref: ColumnRef | None, aggregation: str
) -> str:
    if chart_type == "histogram":
        return f"Distribution of {x_ref.name}"
    if y_ref is None:
        return f"{chart_type.title()} of {x_ref.name}"
    if chart_type == "scatter":
        return f"{y_ref.name} vs {x_ref.name}"
    return f"{aggregation.title()} of {y_ref.name} by {x_ref.name}"
