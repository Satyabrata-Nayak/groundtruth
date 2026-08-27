"""Tests for the benchmark's SQL extractor.

These exist because section D scored 0/10 on a run where the model's SQL was actually
correct — the grader could not find it inside the model's prose. A benchmark whose
harness is wrong is worse than no benchmark: it produces confident, false numbers.

Each case below is a real response shape observed from qwen3:4b.
"""

import sys
from pathlib import Path

import duckdb
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from bench_model import SCHEMA_DDL, strip_sql  # noqa: E402

TOTAL_REVENUE = 7075.0


@pytest.fixture
def con():
    c = duckdb.connect(":memory:")
    c.execute(SCHEMA_DDL)
    yield c
    c.close()


@pytest.mark.parametrize(
    ("name", "response", "expect_salvage"),
    [
        ("bare", "SELECT SUM(revenue) FROM sales", False),
        ("bare_semicolon", "SELECT SUM(revenue) FROM sales;", False),
        ("fenced", "```sql\nSELECT SUM(revenue) FROM sales;\n```", True),
        (
            # The failure that produced 0/10: think=False floods `content` with prose.
            "prose_around_sql",
            "Okay, let's see. The user wants total revenue.\n\n"
            "The query should be SELECT SUM(revenue) FROM sales; \n\n"
            "Wait, let me check whether joins are needed.",
            True,
        ),
        (
            # 'select' as an English verb must not be mistaken for the keyword.
            "prose_uses_word_select",
            "I will select the revenue column here. SELECT SUM(revenue) FROM sales",
            True,
        ),
    ],
)
def test_extracts_executable_sql(con, name, response, expect_salvage):
    sql, salvaged = strip_sql(response)
    assert sql, f"{name}: nothing extracted"
    assert salvaged is expect_salvage, f"{name}: salvage flag wrong"
    assert con.execute(sql).fetchone()[0] == pytest.approx(TOTAL_REVENUE)


def test_cte_is_not_truncated(con):
    """Regression: taking the LAST SELECT|WITH decapitates a CTE.

    `WITH t AS (SELECT ...) SELECT MAX(r) FROM t` — the last keyword is the *inner*
    SELECT, so a naive extractor returns `SELECT MAX(r) FROM t`, which references a CTE
    that no longer exists. A correct query would be scored as a failure.
    """
    response = (
        "First let me think about this. "
        "WITH t AS (SELECT category, SUM(revenue) r FROM sales GROUP BY category) "
        "SELECT MAX(r) FROM t;"
    )
    sql, _ = strip_sql(response)
    assert sql.upper().startswith("WITH"), f"CTE was truncated: {sql!r}"
    assert con.execute(sql).fetchone()[0] == pytest.approx(4555.0)


def test_no_sql_returns_empty():
    """A refusal must be reported as 'no SQL', never as a malformed query."""
    sql, salvaged = strip_sql("I cannot answer that question.")
    assert sql == ""
    assert salvaged is True
