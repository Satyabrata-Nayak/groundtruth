"""Choosing a chart type from the shape of a result.

The complaint that produced this file was "it only ever gives horizontal bars", and the
worse half of the same complaint was a bar chart under an answer that was a single
number. Both are decisions about shape, and the shape is fully known by the time these
functions run — so the choice is made here rather than paid for with a model turn.
"""

from __future__ import annotations

from app.agent.charts import choose_chart
from app.agent.evidence import _chart_from_table


def chart(columns, rows, question="what does this show?"):
    return _chart_from_table({"columns": columns, "rows": rows}, question)


# ============================================================== nothing to draw


def test_a_single_number_gets_no_chart():
    """The correlation case, verbatim: -0.0012 in a one-cell table, under a sentence
    that already said it, with a bar chart of one bar beside it."""
    assert choose_chart(["corr"], [[-0.0012]], "is there a relationship?") is None
    assert chart(["corr"], [[-0.0012]]) is None


def test_a_result_with_no_numbers_gets_no_chart():
    assert choose_chart(["a", "b"], [["x", "y"], ["p", "q"]], "?") is None


# ============================================================== the types


def test_a_ranking_of_categories_is_bars():
    rows = [["UK", 8187806.36], ["Netherlands", 284661.54], ["EIRE", 263276.82]]
    assert choose_chart(["Country", "Revenue"], rows, "which country earns most?") == "bar"


def test_a_long_ranking_stays_bars_rather_than_becoming_a_line():
    """A line between eighty product names draws a trend across things with no order.
    A busy bar chart is honest; a tidy line chart of categories is not."""
    rows = [[f"product-{i}", float(i)] for i in range(80)]
    assert choose_chart(["Product", "Revenue"], rows, "top products?") == "bar"


def test_a_month_column_is_a_line_even_though_it_is_numeric():
    """The ordering bug this test exists for: `Month` and `Revenue` are BOTH numeric, so
    a 'two numeric columns' test that runs before the time test picks scatter — and a
    monthly revenue series came out as a scatter plot."""
    rows = [[month, month * 1000.0] for month in range(1, 13)]
    assert choose_chart(["Month", "Revenue"], rows, "revenue by month?") == "line"


def test_a_camelcase_date_column_is_recognised_as_time():
    """`InvoiceDate` must match and `update` must not, which a substring search cannot
    do. The name is split into words first."""
    rows = [[f"2011-{month:02d}", month * 5.0] for month in range(1, 10)]
    assert choose_chart(["InvoiceDate", "Revenue"], rows, "trend?") == "line"


def test_a_column_that_merely_contains_a_time_word_is_not_time():
    """`year_revenue` holds money, not years. The name matches; the values do not."""
    rows = [["A", 4_000_000.0], ["B", 3_500_000.0], ["C", 900_000.0]]
    assert choose_chart(["Segment", "year_revenue"], rows, "by segment?") == "bar"


def test_a_few_shares_of_a_whole_are_a_pie():
    rows = [["A", 40.0], ["B", 35.0], ["C", 25.0]]
    assert choose_chart(["Segment", "RevenueShare"], rows, "revenue by segment?") == "pie"


def test_a_question_about_shares_is_a_pie_even_without_a_share_column():
    rows = [["A", 4.0], ["B", 3.0], ["C", 2.0]]
    assert choose_chart(["Product", "Revenue"], rows, "what share of revenue?") == "pie"


def test_negative_values_are_never_a_pie():
    """A negative value has no angle. Drawing one as a slice is not a rendering
    limitation, it is a false picture."""
    rows = [["A", -5.0], ["B", 3.0], ["C", 2.0]]
    assert choose_chart(["Region", "ProfitShare"], rows, "share of profit?") == "bar"


def test_too_many_slices_are_never_a_pie():
    rows = [[f"seg-{i}", 10.0] for i in range(15)]
    assert choose_chart(["Segment", "RevenueShare"], rows, "share by segment?") == "bar"


def test_two_numeric_columns_with_no_label_are_a_scatter():
    rows = [[float(i), i * 2.5] for i in range(20)]
    assert choose_chart(["Quantity", "UnitPrice"], rows, "quantity against price?") == "scatter"


def test_one_numeric_column_is_a_distribution():
    rows = [[float(i)] for i in range(40)]
    assert choose_chart(["OrderValue"], rows, "distribution of order value?") == "histogram"


def test_too_few_points_for_a_scatter_falls_back_to_bars():
    rows = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
    assert choose_chart(["a", "b"], rows, "?") == "bar"


# ============================================================== the rendered spec


def test_the_histogram_buckets_the_values_it_was_given():
    """Bucketed from the rows already in hand rather than by asking the database again
    for something we are holding."""
    spec = chart(["OrderValue"], [[float(i)] for i in range(40)])["chart"]
    assert spec["type"] == "histogram"
    assert sum(point["y"] for point in spec["data"]) == 40
    assert spec["y"]["label"] == "rows"


def test_the_maximum_value_lands_inside_the_last_bucket():
    """Off-by-one: the highest value sits exactly on the top edge and indexes one past
    the end of the bucket list."""
    spec = chart(["v"], [[float(i)] for i in range(20)])["chart"]
    assert sum(point["y"] for point in spec["data"]) == 20


def test_a_scatter_carries_both_axes_as_numbers():
    spec = chart(["Quantity", "UnitPrice"], [[float(i), i * 2.0] for i in range(20)])["chart"]
    assert spec["type"] == "scatter"
    assert all(isinstance(point["x"], float) for point in spec["data"])
    assert spec["x"]["kind"] == "numeric"


def test_a_flat_column_gets_no_histogram():
    """Every value identical means zero width, which is a division by zero and a chart
    of nothing."""
    assert chart(["v"], [[5.0] for _ in range(20)]) is None
