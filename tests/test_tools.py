"""The six built-in tools against real data.

These run WITHOUT Postgres. The tools degrade to schema-only when the metadata store
is unreachable, and exercising that path here means the suite stays runnable when
Docker is not, while also proving the degraded mode actually works rather than
assuming it.
"""

from __future__ import annotations

import pytest

from app.data import ingest
from app.tools import ToolContext, get_registry

SALES_CSV = (
    "order_id,order_date,region,category,revenue,cost,units,note\n"
    "1,2024-01-15,North,Widgets,1200.0,800.0,4,ok\n"
    "2,2024-01-20,South,Gadgets,450.0,300.0,2,ok\n"
    "3,2024-02-10,North,Widgets,980.0,700.0,3,\n"
    "4,2024-02-14,East,Doodads,1500.0,410.0,7,ok\n"
    "5,2024-03-02,West,Gadgets,300.0,250.0,1,ok\n"
    "6,2024-03-19,North,Doodads,2100.0,900.0,9,ok\n"
    "7,2024-04-01,South,Widgets,760.0,520.0,3,\n"
    "8,2024-04-22,East,Gadgets,1850.0,1100.0,6,ok\n"
)


@pytest.fixture
def context(tmp_path, data_root):
    source = tmp_path / "sales.csv"
    source.write_text(SALES_CSV, encoding="utf-8")
    result = ingest.ingest_file(source)
    return ToolContext(dataset_id=result.dataset_id, version=result.version)


@pytest.fixture
def wide_context(tmp_path, data_root):
    """A table big enough for the identifier heuristic to apply (>= 20 rows)."""
    source = tmp_path / "wide.csv"
    rows = ["row_id,bucket,value"]
    rows += [f"{i},{'abc'[i % 3]},{i * 10}" for i in range(30)]
    source.write_text("\n".join(rows), encoding="utf-8")
    result = ingest.ingest_file(source)
    return ToolContext(dataset_id=result.dataset_id, version=result.version)


@pytest.fixture
def registry():
    return get_registry()


def call(registry, name, context, **arguments):
    return registry.call(name, context, arguments)


# ------------------------------------------------------------------- inspect_schema


def test_inspect_schema_lists_every_column(registry, context):
    result = call(registry, "inspect_schema", context)
    assert result.ok
    names = [c["name"] for c in result.data["columns"]]
    assert names == [
        "order_id",
        "order_date",
        "region",
        "category",
        "revenue",
        "cost",
        "units",
        "note",
    ]
    assert result.data["row_count"] == 8


def test_inspect_schema_classifies_kinds(registry, context):
    result = call(registry, "inspect_schema", context)
    kinds = {c["name"]: c["kind"] for c in result.data["columns"]}
    assert kinds["revenue"] == "numeric"
    assert kinds["order_date"] == "temporal"
    assert kinds["region"] == "categorical"


def test_inspect_schema_reports_degraded_mode_explicitly(registry, context):
    """Without a stored profile the tool must SAY so, not quietly omit null counts."""
    result = call(registry, "inspect_schema", context)
    assert result.data["profile_available"] is False
    assert "statistics are unavailable" in result.data["note"]


# ------------------------------------------------------------------- profile_column


def test_profile_column_returns_value_frequencies(registry, context):
    result = call(registry, "profile_column", context, column="region")
    assert result.ok
    counts = {entry["value"]: entry["count"] for entry in result.data["top_values"]}
    assert counts == {"North": 3, "South": 2, "East": 2, "West": 1}


def test_profile_column_accepts_wrong_casing(registry, context):
    """Small models get casing wrong constantly; the canonical name is what proceeds."""
    result = call(registry, "profile_column", context, column="REGION")
    assert result.ok
    assert result.data["column"] == "region"


def test_profile_column_suggests_a_correction(registry, context):
    result = call(registry, "profile_column", context, column="reveune")
    assert not result.ok
    assert "Did you mean: revenue?" in result.error


def test_profile_column_unknown_lists_available(registry, context):
    result = call(registry, "profile_column", context, column="zzz")
    assert not result.ok
    assert "Available columns:" in result.error
    assert "revenue" in result.error


# ---------------------------------------------------------------------- execute_sql


def test_execute_sql_aggregates(registry, context):
    result = call(
        registry,
        "execute_sql",
        context,
        sql="SELECT region, sum(revenue) AS r FROM dataset GROUP BY 1 ORDER BY r DESC",
    )
    assert result.ok
    assert result.data["columns"] == ["region", "r"]
    assert result.data["rows"][0] == ["North", 4280.0]


def test_execute_sql_rejects_non_select(registry, context):
    result = call(registry, "execute_sql", context, sql="DROP TABLE dataset")
    assert not result.ok
    assert "only SELECT queries are permitted" in result.error


def test_execute_sql_rejects_reading_other_files(registry, context):
    result = call(registry, "execute_sql", context, sql="SELECT * FROM read_csv('/etc/passwd')")
    assert not result.ok
    assert "table functions are not permitted" in result.error


def test_execute_sql_caps_rows_and_says_so(registry, context):
    result = call(registry, "execute_sql", context, sql="SELECT * FROM dataset", max_rows=3)
    assert result.ok
    assert result.data["row_count"] == 3
    assert result.data["truncated"] is True
    assert "GROUP BY" in result.data["note"]


# -------------------------------------------------------------------- compare_groups


def test_compare_groups_ranks_and_counts(registry, context):
    result = call(
        registry,
        "compare_groups",
        context,
        group_column="region",
        metric_column="revenue",
        aggregation="sum",
    )
    assert result.ok
    top = result.data["groups"][0]
    assert top["group"] == "North"
    assert top["value"] == 4280.0
    assert top["row_count"] == 3
    assert 0 < top["share_of_total"] < 1


def test_compare_groups_withholds_shares_for_averages(registry, context):
    """The average of averages is not the overall average, so a share would mislead."""
    result = call(
        registry,
        "compare_groups",
        context,
        group_column="region",
        metric_column="revenue",
        aggregation="avg",
    )
    assert result.ok
    assert "share_of_total" not in result.data["groups"][0]
    assert "total_across_returned_groups" not in result.data


def test_compare_groups_rejects_text_metric(registry, context):
    result = call(
        registry, "compare_groups", context, group_column="region", metric_column="category"
    )
    assert not result.ok
    assert "needs a numeric column" in result.error
    assert "revenue" in result.error


def test_compare_groups_allows_count_on_any_type(registry, context):
    result = call(
        registry,
        "compare_groups",
        context,
        group_column="region",
        metric_column="category",
        aggregation="count",
    )
    assert result.ok


def test_compare_groups_allows_unique_grouping_in_a_tiny_table(registry, context):
    """Below 20 rows, "every value is distinct" is not evidence of an identifier.

    This fixture has 8 rows and 8 order_ids. Calling that a high-cardinality column
    would refuse a legitimate grouping on any small dataset, so the rule deliberately
    does not apply here -- see `_IDENTIFIER_MIN_ROWS`.
    """
    result = call(
        registry, "compare_groups", context, group_column="order_id", metric_column="revenue"
    )
    assert result.ok


def test_compare_groups_rejects_identifier_grouping(registry, wide_context):
    """With enough rows, a column that is unique per row aggregates nothing."""
    result = call(
        registry,
        "compare_groups",
        context=wide_context,
        group_column="row_id",
        metric_column="value",
    )
    assert not result.ok
    assert "one group per row" in result.error
    assert "identifier" in result.error


def test_compare_groups_accepts_a_real_category_in_the_same_table(registry, wide_context):
    """The rejection above must be about uniqueness, not about the table being large."""
    result = call(
        registry,
        "compare_groups",
        context=wide_context,
        group_column="bucket",
        metric_column="value",
    )
    assert result.ok
    assert len(result.data["groups"]) == 3


# ----------------------------------------------------------------------- correlation


def test_correlation_reports_both_coefficients(registry, context):
    result = call(registry, "correlation", context, column_a="revenue", column_b="cost")
    assert result.ok
    assert result.data["rows_used"] == 8
    assert result.data["pearson"] is not None
    assert result.data["spearman"] is not None
    assert result.data["direction"] == "positive"


def test_correlation_rejects_text_column(registry, context):
    result = call(registry, "correlation", context, column_a="region", column_b="revenue")
    assert not result.ok
    assert "needs a numeric column" in result.error


def test_correlation_rejects_a_column_against_itself(registry, context):
    result = call(registry, "correlation", context, column_a="revenue", column_b="revenue")
    assert not result.ok
    assert "correlates" in result.error


# ---------------------------------------------------------------------- create_chart


def test_bar_chart_aggregates_per_category(registry, context):
    result = call(registry, "create_chart", context, chart_type="bar", x="region", y="revenue")
    assert result.ok
    chart = result.data["chart"]
    assert chart["type"] == "bar"
    assert chart["point_count"] == 4
    assert chart["data"][0] == {"x": "North", "y": 4280.0}


def test_line_chart_refuses_an_unordered_x_axis(registry, context):
    result = call(registry, "create_chart", context, chart_type="line", x="region", y="revenue")
    assert not result.ok
    assert "temporal or numeric" in result.error


def test_histogram_covers_every_bucket(registry, context):
    result = call(registry, "create_chart", context, chart_type="histogram", x="revenue", bins=4)
    assert result.ok
    data = result.data["chart"]["data"]
    assert len(data) == 4  # empty buckets still appear
    assert sum(point["count"] for point in data) == 8


def test_histogram_refuses_a_y_column(registry, context):
    result = call(registry, "create_chart", context, chart_type="histogram", x="revenue", y="cost")
    assert not result.ok
    assert "does not use a 'y' column" in result.error


def test_chart_model_view_is_smaller_than_the_render_payload(registry, context):
    result = call(registry, "create_chart", context, chart_type="scatter", x="revenue", y="cost")
    assert result.ok
    assert result.data["chart"]["point_count"] == 8
    # Eight points is under the threshold, so nothing is trimmed here -- the contract
    # under test is that the caller's payload is never reduced.
    assert len(result.data["chart"]["data"]) == 8


def test_chart_titles_describe_the_chart(registry, context):
    result = call(registry, "create_chart", context, chart_type="bar", x="region", y="revenue")
    assert result.data["chart"]["title"] == "Sum of revenue by region"


def test_unknown_chart_type_is_rejected_by_the_schema(registry, context):
    result = call(registry, "create_chart", context, chart_type="pie", x="region", y="revenue")
    assert not result.ok
    assert "must be one of" in result.error
