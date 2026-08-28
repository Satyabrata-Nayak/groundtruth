"""Profile correctness, checked against hand-computed values.

Every expected number here was worked out by hand from the fixture below, not copied
from the code's own output. A test that asserts the implementation agrees with itself
proves nothing.

    revenue: 100, 200, 300, NULL     sum 600, mean 200, min 100, max 300
    region:  North, South, North, North   3 distinct incl. NULL? no -> 2 distinct
    flag:    always 'X'              constant
    note:    all different           high cardinality (small n, so NOT flagged)
"""

import pytest

from app.config import get_settings
from app.data import ingest
from app.data.profile import classify_type, profile_parquet


@pytest.fixture
def data_root(tmp_path, monkeypatch):
    root = tmp_path / "datasets"
    root.mkdir()
    monkeypatch.setattr(get_settings(), "data_dir", root)
    return root


@pytest.fixture
def profiled(data_root, tmp_path):
    csv = tmp_path / "fixture.csv"
    csv.write_text(
        "order_id,order_date,region,revenue,flag,note\n"
        "1,2024-01-15,North,100.0,X,alpha\n"
        "2,2024-02-20,South,200.0,X,beta\n"
        "3,2024-03-10,North,300.0,X,gamma\n"
        "4,2024-04-05,North,,X,delta\n",
        encoding="utf-8",
    )
    result = ingest.ingest_file(csv)
    return profile_parquet(result.parquet_path)


def col(profile, name):
    return next(c for c in profile.columns if c.name == name)


# --------------------------------------------------------------------------------
# Shape
# --------------------------------------------------------------------------------


def test_shape(profiled):
    assert profiled.row_count == 4
    assert profiled.column_count == 6
    assert len(profiled.columns) == 6


def test_column_order_is_preserved(profiled):
    """Position must match the file. The UI and the agent both present columns in
    file order, and a reordered profile silently mislabels everything."""
    assert [c.name for c in profiled.columns] == [
        "order_id",
        "order_date",
        "region",
        "revenue",
        "flag",
        "note",
    ]
    assert [c.position for c in profiled.columns] == [0, 1, 2, 3, 4, 5]


# --------------------------------------------------------------------------------
# Type classification
# --------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("duckdb_type", "expected"),
    [
        ("BIGINT", "numeric"),
        ("INTEGER", "numeric"),
        ("DOUBLE", "numeric"),
        ("DECIMAL(10,2)", "numeric"),
        ("FLOAT", "numeric"),
        ("DATE", "temporal"),
        ("TIMESTAMP", "temporal"),
        ("TIMESTAMP WITH TIME ZONE", "temporal"),
        ("BOOLEAN", "boolean"),
        ("VARCHAR", "categorical"),
        ("BLOB", "categorical"),
    ],
)
def test_type_classification(duckdb_type, expected):
    assert classify_type(duckdb_type) == expected


def test_semantic_types_on_real_data(profiled):
    assert col(profiled, "order_id").semantic_type == "numeric"
    assert col(profiled, "order_date").semantic_type == "temporal"
    assert col(profiled, "region").semantic_type == "categorical"
    assert col(profiled, "revenue").semantic_type == "numeric"


# --------------------------------------------------------------------------------
# Statistics — hand-computed
# --------------------------------------------------------------------------------


def test_null_counts_are_exact(profiled):
    """Exact, not SUMMARIZE's rounded percentage. Rounding is fine for display and
    wrong for arithmetic — 0.4% of 2.4M rows rounds to a figure off by thousands."""
    assert col(profiled, "revenue").null_count == 1
    assert col(profiled, "revenue").null_fraction == pytest.approx(0.25)
    assert col(profiled, "region").null_count == 0
    assert col(profiled, "region").null_fraction == 0.0


def test_numeric_statistics(profiled):
    revenue = col(profiled, "revenue")
    # mean of 100, 200, 300 — nulls excluded, which is what SQL AVG does
    assert revenue.mean_value == pytest.approx(200.0)
    assert float(revenue.min_value) == pytest.approx(100.0)
    assert float(revenue.max_value) == pytest.approx(300.0)


def test_text_column_has_no_numeric_stats(profiled):
    """'mean' of a text column is not 0, it is undefined. NULL keeps 'not applicable'
    distinguishable from 'genuinely zero'."""
    region = col(profiled, "region")
    assert region.mean_value is None
    assert region.stddev_value is None
    assert region.q50_value is None


def test_distinct_counts(profiled):
    assert col(profiled, "region").distinct_count == 2  # North, South
    assert col(profiled, "flag").distinct_count == 1
    assert col(profiled, "order_id").distinct_count == 4


# --------------------------------------------------------------------------------
# Quality flags
# --------------------------------------------------------------------------------


def test_constant_column_is_flagged(profiled):
    """A single-valued column cannot explain variation in anything."""
    assert col(profiled, "flag").is_constant is True
    assert col(profiled, "region").is_constant is False


def test_duplicate_rows_counted_as_removable(data_root, tmp_path):
    """N identical rows = N-1 duplicates: the number you would delete to deduplicate."""
    csv = tmp_path / "dupes.csv"
    csv.write_text("a,b\n1,x\n1,x\n1,x\n2,y\n", encoding="utf-8")
    profile = profile_parquet(ingest.ingest_file(csv).parquet_path)

    assert profile.row_count == 4
    assert profile.duplicate_row_count == 2


def test_no_duplicates_reports_zero(profiled):
    assert profiled.duplicate_row_count == 0


def test_high_cardinality_needs_enough_rows(data_root, tmp_path):
    """With 4 rows every text column looks unique. The flag means 'behaves like an
    identifier', which cannot be judged on a tiny sample — so it requires a minimum
    row count and would otherwise fire on every small dataset."""
    csv = tmp_path / "ids.csv"
    rows = "\n".join(f"{i},user_{i}" for i in range(1, 31))
    csv.write_text(f"id,username\n{rows}\n", encoding="utf-8")
    profile = profile_parquet(ingest.ingest_file(csv).parquet_path)

    assert col(profile, "username").is_high_cardinality is True


def test_low_cardinality_not_flagged(data_root, tmp_path):
    csv = tmp_path / "cats.csv"
    rows = "\n".join(f"{i},{'North' if i % 2 else 'South'}" for i in range(1, 31))
    csv.write_text(f"id,region\n{rows}\n", encoding="utf-8")
    profile = profile_parquet(ingest.ingest_file(csv).parquet_path)

    assert col(profile, "region").is_high_cardinality is False


# --------------------------------------------------------------------------------
# Hostile column names
# --------------------------------------------------------------------------------


def test_awkward_column_names_are_handled(data_root, tmp_path):
    """Column names come from an uploaded file. Spaces, quotes, SQL keywords and
    non-ASCII all have to survive being interpolated into the profiling queries."""
    csv = tmp_path / "awkward.csv"
    csv.write_text(
        'normal,"has space","has""quote",select,café\n1,2,3,4,5\n6,7,8,9,10\n',
        encoding="utf-8",
    )
    profile = profile_parquet(ingest.ingest_file(csv).parquet_path)

    names = [c.name for c in profile.columns]
    assert "has space" in names
    assert 'has"quote' in names
    assert "select" in names
    assert "café" in names
    assert all(c.null_count == 0 for c in profile.columns)


def test_total_null_fraction(profiled):
    """1 null cell out of 4 rows x 6 columns = 24 cells."""
    assert profiled.total_null_fraction == pytest.approx(1 / 24)


def test_distinct_counts_are_exact_not_approximate(data_root, tmp_path):
    """Regression: SUMMARIZE's `approx_unique` is HyperLogLog, not a count.

    On this exact fixture — 30 rows, 30 genuinely distinct usernames — `approx_unique`
    returns 27, a 10% error. That was enough to flip the high-cardinality threshold and
    would have put an estimate in a field named `distinct_count`.
    """
    csv = tmp_path / "exact.csv"
    rows = "\n".join(f"{i},user_{i}" for i in range(1, 31))
    csv.write_text(f"id,username\n{rows}\n", encoding="utf-8")
    profile = profile_parquet(ingest.ingest_file(csv).parquet_path)

    assert col(profile, "username").distinct_count == 30
    assert col(profile, "id").distinct_count == 30


def test_distinct_count_excludes_nulls(data_root, tmp_path):
    """SQL semantics: count(DISTINCT x) ignores NULL. Documented so a reader does not
    have to guess whether NULL counts as its own value."""
    csv = tmp_path / "nulls.csv"
    csv.write_text("a\nx\ny\n\nx\n", encoding="utf-8")
    profile = profile_parquet(ingest.ingest_file(csv).parquet_path)

    column = col(profile, "a")
    assert column.distinct_count == 2  # x, y — not 3
    assert column.null_count == 1
