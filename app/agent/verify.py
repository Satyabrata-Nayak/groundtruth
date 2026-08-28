"""Checking the answer's numbers against the numbers that were actually computed.

WHY THIS EXISTS
---------------
The first end-to-end run on real data produced this, and it is worth quoting because
everything about it looks right:

    "WORLD WAR 2 GLIDERS ASSTD DESIGNS with 53,847 units, significantly exceeding the
     next highest product JUMBO BAG RED RETROSPOT by 16,484 units"

53,847 and 47,363 both came from DuckDB. 16,484 came from the model's head, and the
correct difference is 6,484. Every guard in the loop was working: the schema was real,
the SQL was real, the table underneath the sentence was real. The model still asserted
a number nobody computed, in a sentence where it reads as the most natural thing in
the world.

That is the whole failure mode this project was built to prevent, and no amount of
prompt wording removes it — a language model does arithmetic by autocomplete, and
autocomplete is right most of the time.

WHAT THIS DOES AND WHAT IT DELIBERATELY DOES NOT DO
---------------------------------------------------
It pulls every figure out of the answer, pulls every number out of every successful
tool result, and reports the figures that cannot be traced to one. It does NOT rewrite
the answer, and it does NOT fail the analysis. It attaches a warning naming the exact
untraceable figures, which the UI shows above the evidence.

Refusing to silently "fix" the sentence is the point. A number the system cannot trace
is a fact about the system's confidence, and hiding it — by dropping the sentence, or
by quietly correcting it — would be a different way of asserting more than is known.

WHY THE MATCHING IS LOOSE
-------------------------
An answer says "8,187,806.36" for a stored 8187806.363998184, "35%" for a stored share
of 0.3528, and "1.2 million" for neither. So a figure matches if it equals a computed
number, is that number rounded to any sensible number of places, or is within a small
relative tolerance of it. Fractions are also matched against their percentage form,
because `share_of_total` is a fraction and every human writes it as a percent.

The bias is deliberately towards NOT warning: a warning on a correct answer teaches
people to ignore warnings, and then the one that matters is ignored too.
"""

from __future__ import annotations

import re
from typing import Any

from app.tools.base import ToolResult

# Figures below this are skipped. "the top 10 countries", "3 groups", "2011" and "one
# row per customer" are all numbers in a sentence that no tool needs to have produced,
# and flagging them would bury the one figure that matters under noise.
_MIN_INTERESTING = 100.0

# How many untraceable figures to name before it stops being a warning and starts being
# a wall of text. If there are more than this, the answer has bigger problems.
_MAX_REPORTED = 4

# Relative tolerance for "this is the same number, written differently".
_RELATIVE_TOLERANCE = 0.005

# A number as a person writes one: 53,847 · 8187806.36 · -12.5 · 1000
_FIGURE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def untraceable_figures(answer: str, results: list[ToolResult]) -> list[str]:
    """Figures in `answer` that match no number in any successful tool result."""
    if not answer.strip():
        return []

    known = _computed_numbers(results)
    if not known:
        return []

    unmatched: list[str] = []
    for text, value in _figures_in(answer):
        if any(_matches(value, candidate) for candidate in known):
            continue
        if text not in unmatched:
            unmatched.append(text)
    return unmatched[:_MAX_REPORTED]


def verification_warning(answer: str, results: list[ToolResult]) -> str | None:
    """The one-line warning, or None when every figure traces back to a computation."""
    unmatched = untraceable_figures(answer, results)
    if not unmatched:
        return None
    figures = ", ".join(unmatched)
    plural = "figures do" if len(unmatched) > 1 else "figure does"
    return (
        f"the following {plural} not appear in any tool result and may have been "
        f"calculated by the model rather than by the database: {figures}"
    )


def _figures_in(answer: str) -> list[tuple[str, float]]:
    """Every figure worth checking, as (as written, as a number)."""
    found: list[tuple[str, float]] = []
    for match in _FIGURE.finditer(answer):
        text = match.group()
        try:
            value = float(text.replace(",", ""))
        except ValueError:  # pragma: no cover - the regex cannot produce this
            continue
        if abs(value) < _MIN_INTERESTING:
            continue
        found.append((text, value))
    return found


def _computed_numbers(results: list[ToolResult]) -> list[float]:
    """Every number any successful tool produced, plus percentage forms of fractions.

    `data` is walked rather than a known set of keys read, because every tool has a
    different payload shape and a checker that only understood two of them would pass
    an answer built on the third.
    """
    numbers: list[float] = []
    for result in results:
        if result.ok:
            _collect(result.data, numbers)

    # `share_of_total` is a fraction and every answer writes it as a percent.
    #
    # Materialised into its own list first. `numbers.extend(v * 100 for v in numbers)`
    # reads correctly and hangs forever: the generator iterates the same list extend is
    # appending to, and a single 0.0 in the data (0.0 * 100 == 0.0, still in range)
    # regenerates itself without end. It hung the whole test suite.
    percentages = [value * 100 for value in numbers if 0.0 <= value <= 1.0]
    return numbers + percentages


def _collect(value: Any, into: list[float]) -> None:
    if isinstance(value, bool):
        return
    if isinstance(value, (int, float)):
        into.append(float(value))
    elif isinstance(value, dict):
        for item in value.values():
            _collect(item, into)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _collect(item, into)
    elif isinstance(value, str):
        # Numbers inside strings count: a summary line reading "top: West = 3255" is a
        # computed result, and an answer quoting it is quoting the database.
        for match in _FIGURE.finditer(value):
            try:
                into.append(float(match.group().replace(",", "")))
            except ValueError:  # pragma: no cover
                continue


def _matches(written: float, computed: float) -> bool:
    """Is `written` a reasonable way of writing `computed`?"""
    if written == computed:
        return True
    for places in range(5):
        if round(computed, places) == written:
            return True
    if computed != 0 and abs(written - computed) / abs(computed) <= _RELATIVE_TOLERANCE:
        return True
    return False
