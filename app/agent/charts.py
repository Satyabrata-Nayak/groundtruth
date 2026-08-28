"""Choosing the right chart for a result, from the shape of the result.

WHY THIS IS DECIDED HERE AND NOT BY THE MODEL
---------------------------------------------
Asking the model which chart to draw costs a turn, and a turn is 45-90 seconds. The
chart type is also not really a judgement call: it is a function of the result's shape,
and the result is sitting right here, fully known, with its types and its row count.
Deciding it in code takes microseconds and cannot pick `pie` for four hundred rows.

    "which country earns most"      10 labels, 1 value      -> bar
    "revenue by month"              dates in order          -> line
    "share of revenue by segment"   <= 8 slices, all + ve   -> pie
    "quantity against unit price"   two numeric columns     -> scatter
    "distribution of order value"   one numeric column      -> histogram
    "the correlation is -0.001"     a single number         -> NO CHART

The last row matters as much as the others. A one-row result has nothing to compare
against, and a bar chart of one bar — or worse, a table with one cell — is decoration
presented as analysis. The honest output is no chart at all.

WHAT THE MODEL STILL CONTROLS
-----------------------------
The shape. It is told in the system prompt that a date column gives a line and a small
set of shares gives a pie, so "show me the monthly trend" produces a query whose result
IS a trend, and the trend gets drawn as one. That is a better division than asking it
to name a chart type: it chooses what to compute, and the computation decides how it
looks — the same rule the rest of this system runs on.
"""

from __future__ import annotations

import re
from typing import Any

# A pie is only readable with few slices, and only honest when the parts are of one
# whole: negative values have no angle, and twenty slices is a colour-matching puzzle.
_MAX_PIE_SLICES = 8
# A scatter needs enough points to show a relationship rather than three dots.
_MIN_SCATTER_POINTS = 12
_MIN_HISTOGRAM_POINTS = 12

# Words that mean "this axis is time". Matched against the words OF THE NAME rather
# than as a substring: `InvoiceDate` must match and `update` must not, and a plain
# substring search cannot tell those apart. The name is the signal because the value
# may be a VARCHAR ('12/1/10 8:26'), an integer (a month number, a year) or a real
# date — and in the first two cases the type says nothing about intent.
_TIME_WORDS = frozenset(
    {
        "date",
        "dates",
        "day",
        "days",
        "month",
        "months",
        "year",
        "years",
        "week",
        "weeks",
        "quarter",
        "quarters",
        "hour",
        "hours",
        "minute",
        "time",
        "period",
        "periods",
        "dt",
        "ts",
        "timestamp",
        "yearmonth",
        "datetime",
    }
)

# ...and words that mean "this value is a share of a whole", which is what a pie needs.
_SHARE_WORDS = frozenset(
    {"share", "percent", "percentage", "pct", "proportion", "ratio", "fraction"}
)

# Splits `InvoiceDate`, `invoice_date` and `INVOICE DATE` alike into their words.
_WORDS = re.compile(r"[A-Z]+(?![a-z])|[A-Z][a-z]+|[a-z]+|[0-9]+")


def _words(name: str) -> set[str]:
    return {word.lower() for word in _WORDS.findall(str(name))}


def choose_chart(columns: list[str], rows: list[list[Any]], question: str) -> str | None:
    """The chart type for this result, or None when a chart would add nothing.

    The order of these tests is the whole design, and getting it wrong is how a monthly
    revenue series came out as a scatter plot: `Month` and `Revenue` are both numeric,
    so a "two numeric columns" test that runs before the time test wins for the wrong
    reason. Time is checked first because a time axis settles the question whatever the
    other columns are.
    """
    if not rows or not columns:
        return None
    # One row is an answer, not a comparison, and a chart of it is decoration.
    if len(rows) < 2:
        return None

    numeric = [index for index in range(len(columns)) if _is_numeric_column(rows, index)]
    if not numeric:
        return None
    labels = [index for index in range(len(columns)) if index not in numeric]

    # 1. Anything measured against time is a trend.
    for index in range(len(columns)):
        if _looks_like_time(columns[index], rows, index):
            # ...unless time is the only column, which is a distribution of timestamps.
            if len(columns) > 1:
                return "line"

    # 2. Numbers with nothing to label them: one column is a distribution, two are a
    #    relationship. Both need enough points to show anything.
    if not labels:
        if len(numeric) == 1:
            return "histogram" if len(rows) >= _MIN_HISTOGRAM_POINTS else None
        return "scatter" if len(rows) >= _MIN_SCATTER_POINTS else "bar"

    # 3. Parts of a whole: few enough to tell apart, all positive, and either named as
    #    a share or asked about as one. A negative value has no angle.
    if len(rows) <= _MAX_PIE_SLICES:
        share_index = next((i for i in numeric if _words(columns[i]) & _SHARE_WORDS), None)
        if share_index is not None and _all_positive(rows, share_index):
            return "pie"
        if _asks_about_share(question) and _all_positive(rows, numeric[0]):
            return "pie"

    # 4. A labelled ranking. Bars, however many: a line between product names would
    #    draw a trend across categories that have no order, which is a lie about the
    #    data rather than a busy chart.
    return "bar"


def _is_numeric_column(rows: list[list[Any]], index: int) -> bool:
    seen = False
    for row in rows:
        value = row[index] if index < len(row) else None
        if value is None:
            continue
        # bool is an int in Python, and a column of flags is not a magnitude.
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            return False
        seen = True
    return seen


def _looks_like_time(name: str, rows: list[list[Any]], index: int) -> bool:
    """Is this column an axis of time?

    A name match alone is enough for a text column. For a NUMERIC one the values must
    also be plausible as periods, so `year_revenue` — which contains the word "year"
    and holds figures in the millions — is not mistaken for a time axis.
    """
    if not (_words(name) & _TIME_WORDS):
        return False
    values = [row[index] for row in rows if index < len(row) and row[index] is not None]
    if not values:
        return False
    if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in values):
        # Month numbers, quarters, week numbers, years, and YYYYMM keys.
        return all(float(v).is_integer() and 0 <= float(v) <= 999999 for v in values)
    return True


def _all_positive(rows: list[list[Any]], index: int) -> bool:
    return all(
        row[index] is None or (isinstance(row[index], (int, float)) and row[index] > 0)
        for row in rows
        if index < len(row)
    )


def _asks_about_share(question: str) -> bool:
    return bool(_words(question) & (_SHARE_WORDS | {"breakdown", "split", "composition"}))
