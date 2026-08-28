"""Which analysis engine runs, and the deterministic one that needs no model.

    run_analysis()  ──►  ANALYSIS_ENGINE=agent  ──►  app/agent/analyst.py   (M5)
                    └──►  ANALYSIS_ENGINE=fixed  ──►  run_fixed_analysis()   (M4)

WHY THE FIXED ENGINE SURVIVED M5
--------------------------------
The obvious move was to delete this file once the agent worked. Keeping it is worth
more than the eighty lines it costs, because it is the only way to answer one question
quickly: *is this broken, or is the model just bad at it?* Set `ANALYSIS_ENGINE=fixed`
and the whole stack — API, queue, worker, tools, sandbox, UI — runs with no model in
it. If it still fails, the model was never the problem.

It is also what makes the test suite runnable on a machine with no Ollama, and it is
the calibration floor the eval harness already measures against.

THE SHARED RESULT CONTRACT
--------------------------
    {engine, question, dataset, answer, steps[], table, chart, warnings[]}

is what the API serialises, what the UI renders, and what both engines fill. It was
fixed in M4 for exactly this reason: M5 replaced one function and changed neither the
database, the API nor the frontend.

WHY IT GOES THROUGH THE TOOL REGISTRY
-------------------------------------
It would be shorter to run three SQL strings against the sandbox directly. Going
through `ToolRegistry.call` instead means M4 proves the M3 action space actually
works end to end — argument validation, column resolution, the row caps, the
two-audience payloads — while there is still no model around to confuse the picture.
Every bug found here is a bug M5 does not get to blame on a 4B model.

    inspect_schema  ──►  pick a grouping column and a metric   (the only "choice",
                                                                and it is a rule)
    compare_groups  ──►  the table
    create_chart    ──►  the bar chart
"""

from __future__ import annotations

import uuid
from typing import Any

from app.agent.analyst import run_agent_analysis
from app.agent.contract import AnalysisFailed, Checkpoint, Emit
from app.config import get_settings
from app.db.models import EventKind
from app.tools import get_registry
from app.tools.base import ToolContext, ToolResult

# The result shape is versioned so a stored analysis says what produced it. M5 writes
# "agent-v1" here, and a row from M4 stays interpretable next to it forever.
FIXED_ENGINE = "hardcoded-v1"

# A grouping column with more distinct values than this is an identifier in disguise,
# and grouping by it produces one row per row. `compare_groups` enforces its own,
# stricter rule; this one only decides which column to *offer* it.
_MAX_GROUPS_FOR_DISPLAY = 25

# A numeric column with nearly as many distinct values as there are rows is an
# identifier, whatever it is called. Relative, not absolute, and mirroring the rule
# `app/tools/stats.py` applies to grouping columns.
_IDENTIFIER_FRACTION = 0.9
_IDENTIFIER_MIN_ROWS = 20

# ...but only for integers. A DOUBLE with one distinct value per row is a continuous
# measurement, not a key; nobody stores a primary key as a float.
_INTEGER_TYPES = frozenset(
    {
        "TINYINT",
        "SMALLINT",
        "INTEGER",
        "BIGINT",
        "HUGEINT",
        "UTINYINT",
        "USMALLINT",
        "UINTEGER",
        "UBIGINT",
        "UHUGEINT",
    }
)


def run_analysis(
    *,
    dataset_id: uuid.UUID,
    version: int,
    question: str,
    emit: Emit,
    checkpoint: Checkpoint,
    llm_model: str | None = None,
    llm_thinking: bool | None = None,
) -> dict[str, Any]:
    """Run the configured engine and return the result payload.

    The worker calls only this. Which engine runs is configuration, not a decision made
    at the call site, so switching one off never means editing the worker.

    `llm_model` and `llm_thinking` are what the ASKER chose, pinned on the row when the
    question was queued. The fixed engine ignores them, which is correct: it has no
    model to choose.
    """
    if get_settings().analysis_engine == "fixed":
        emit(EventKind.NOTE, "running the fixed analysis (no model)", {"engine": FIXED_ENGINE})
        return run_fixed_analysis(
            dataset_id=dataset_id,
            version=version,
            question=question,
            emit=emit,
            checkpoint=checkpoint,
        )
    return run_agent_analysis(
        dataset_id=dataset_id,
        version=version,
        question=question,
        emit=emit,
        checkpoint=checkpoint,
        llm_model=llm_model,
        llm_thinking=llm_thinking,
    )


def run_fixed_analysis(
    *,
    dataset_id: uuid.UUID,
    version: int,
    question: str,
    emit: Emit,
    checkpoint: Checkpoint,
) -> dict[str, Any]:
    """Compare the first usable numeric column across the first usable categorical one.

    `emit` records an observable event; `checkpoint` raises if the work should stop.
    Both are passed in rather than imported so this function has no opinion about
    transactions or threads — which is what made the M5 agent a drop-in replacement.
    """
    registry = get_registry()
    context = ToolContext(dataset_id=dataset_id, version=version)
    steps: list[dict[str, Any]] = []

    def call(tool: str, **arguments: Any) -> ToolResult:
        checkpoint()
        emit(EventKind.TOOL_CALL, f"calling {tool}", {"tool": tool, "arguments": arguments})
        result = registry.call(tool, context, arguments)
        steps.append(
            {
                "tool": result.tool,
                "arguments": result.arguments,
                "ok": result.ok,
                "summary": result.summary,
                "error": result.error,
                "duration_ms": round(result.duration_ms, 2),
            }
        )
        emit(
            EventKind.TOOL_RESULT,
            result.summary if result.ok else f"{tool} failed: {result.error}",
            {"tool": result.tool, "ok": result.ok, "duration_ms": round(result.duration_ms, 2)},
        )
        return result

    # --- 1. look at the data before deciding anything about it ----------------
    schema = call("inspect_schema", include_statistics=True)
    if not schema.ok:
        raise AnalysisFailed(f"could not read the dataset schema: {schema.error}")

    columns: list[dict[str, Any]] = schema.data["columns"]
    row_count = schema.data.get("row_count")

    group_column = _pick_group_column(columns)
    metric_column = _pick_metric_column(columns, row_count)

    if group_column is None or metric_column is None:
        # Not a failure: a dataset of pure text, or pure numbers, is a legitimate
        # thing to upload. Answer the question that CAN be answered and say so.
        return _describe_only(question, dataset_id, version, schema, steps, row_count)

    emit(
        EventKind.NOTE,
        f"comparing {metric_column} across {group_column}",
        {"group_column": group_column, "metric_column": metric_column},
    )

    # --- 2. the comparison ----------------------------------------------------
    comparison = call(
        "compare_groups",
        group_column=group_column,
        metric_column=metric_column,
        aggregation="sum",
        order="desc",
        limit=_MAX_GROUPS_FOR_DISPLAY,
    )
    if not comparison.ok:
        raise AnalysisFailed(f"comparison failed: {comparison.error}")

    # --- 3. the picture -------------------------------------------------------
    chart = call(
        "create_chart",
        chart_type="bar",
        x=group_column,
        y=metric_column,
        aggregation="sum",
        title=f"Total {metric_column} by {group_column}",
    )

    return {
        "engine": FIXED_ENGINE,
        "question": question,
        "dataset": {"id": str(dataset_id), "version": version},
        "answer": _write_answer(question, group_column, metric_column, comparison, row_count),
        "steps": steps,
        "table": _table_from(comparison),
        "chart": chart.data if chart.ok else None,
        # Always present, always empty: the fixed engine has no judgement to qualify.
        "warnings": [],
    }


# --------------------------------------------------------------------------------
# Column choice — the only decision, and it is a rule rather than a judgement
# --------------------------------------------------------------------------------


def _pick_group_column(columns: list[dict[str, Any]]) -> str | None:
    """The first categorical column that would actually make a readable chart.

    Skips constants (one bar) and identifiers (one bar per row). `distinct_count` is
    exact — see D-012 — so this threshold means what it says; with the HyperLogLog
    estimate DuckDB offers by default it would silently misjudge columns near the
    boundary.
    """
    for column in columns:
        if column.get("kind") != "categorical":
            continue
        if column.get("warning"):  # constant, or high-cardinality
            continue
        distinct = column.get("distinct_count")
        if distinct is not None and 1 < distinct <= _MAX_GROUPS_FOR_DISPLAY:
            return str(column["name"])
    return None


def _pick_metric_column(columns: list[dict[str, Any]], row_count: int | None) -> str | None:
    """The first numeric column that is a measurement rather than an identifier.

    THE BUG THIS FUNCTION WAS WRITTEN TWICE TO FIX
    ----------------------------------------------
    The first version trusted `inspect_schema`'s `warning` field to flag identifiers.
    It does not: `is_high_cardinality` is only computed for CATEGORICAL columns (see
    `app/data/profile.py`), because a numeric column with many distinct values is
    usually a measurement, not an id. So on the very first real run this picked
    `order_id` and reported "the largest total order_id is in region West at
    4,410,927" — arithmetic that is perfectly correct and completely meaningless.

    ...AND THE BUG THE FIRST FIX INTRODUCED
    ---------------------------------------
    The obvious replacement — "distinct values ≈ row count means identifier" — is
    right for `order_id` and wrong for `revenue`. A continuous measurement over 5,000
    rows also has ~5,000 distinct values, so the rule rejected every float in the
    table and settled on `units`, whose small integer range let it slip through. The
    analysis was no longer meaningless, just quietly worse, which is the harder kind
    of wrong to notice.

    The discriminator is the TYPE. Near-uniqueness only implies an identifier when the
    column is an integer; a DOUBLE with 5,000 distinct values is a measurement, and
    identifiers are essentially never stored as floats.

      structural   INTEGER type AND distinct ≈ rows  ->  identifies rows, measures
                                                         nothing. Decides.
      nominal      the name ends in `id`             ->  a preference, never a veto:
                                                         `bid`, `grid` and `void` all
                                                         end in "id", and a column
                                                         called `customer_number`
                                                         does not.

    A dataset whose only numeric column is an identifier still gets summed — the
    fallback — and the answer says which columns it used, so a reader can see it.
    """
    preferred: str | None = None
    fallback: str | None = None

    for column in columns:
        if column.get("kind") != "numeric" or column.get("warning"):
            continue
        name = str(column["name"])

        if _looks_like_an_identifier(column, row_count):
            continue

        if name.lower().endswith("id"):
            fallback = fallback or name
            continue
        preferred = preferred or name

    return preferred or fallback


def _looks_like_an_identifier(column: dict[str, Any], row_count: int | None) -> bool:
    """An integer column with nearly one distinct value per row.

    Relative, not absolute: 5,000 distinct values is an identifier in a 5,000-row
    table and an ordinary measurement in a 5,000,000-row one. Same argument as
    `_reject_unusable_grouping` in `app/tools/stats.py`, and it relies on
    `distinct_count` being exact rather than a HyperLogLog estimate (D-012) — near the
    threshold, a 10% error decides the answer.
    """
    if str(column.get("type", "")).upper() not in _INTEGER_TYPES:
        return False
    distinct = column.get("distinct_count")
    if distinct is None or not row_count or row_count < _IDENTIFIER_MIN_ROWS:
        return False
    return distinct >= row_count * _IDENTIFIER_FRACTION


# --------------------------------------------------------------------------------
# Presentation
# --------------------------------------------------------------------------------


def _table_from(comparison: ToolResult) -> dict[str, Any]:
    groups: list[dict[str, Any]] = comparison.data.get("groups", [])
    if not groups:
        return {"columns": [], "rows": []}
    headers = list(groups[0])
    return {
        "columns": headers,
        "rows": [[group.get(header) for header in headers] for group in groups],
    }


def _write_answer(
    question: str,
    group_column: str,
    metric_column: str,
    comparison: ToolResult,
    row_count: int | None,
) -> str:
    """A sentence a person can check against the table underneath it.

    Every number in here comes from `comparison.data`, never from this function's own
    arithmetic. That is the same rule the M5 agent will be held to, and stating it in
    the one place where it would be trivially easy to break is the point.
    """
    groups: list[dict[str, Any]] = comparison.data.get("groups", [])
    scale = f" across {row_count:,} rows" if row_count else ""

    if not groups:
        return (
            f"No groups were produced for '{group_column}'{scale}, so there is nothing "
            f"to compare. (This is a fixed M4 analysis, not an answer to your question.)"
        )

    # `compare_groups` keys the group label as "group", NOT as the column name. The
    # first version wrote `top.get(group_column)`, which is always None — so the
    # sentence read "region = None" directly above a table whose first row said
    # "West". Wrong, and worse than wrong: plausible, and contradicted by the evidence
    # printed underneath it. Reading a tool's payload beats assuming its shape.
    top = groups[0]
    lead = (
        f"The largest total {metric_column} is in {group_column} = {top['group']!r} "
        f"at {top.get('value')}"
    )
    share = top.get("share_of_total")
    if share is not None:
        # A FRACTION, not a percentage: `compare_groups` returns value/total. Printing
        # it with a % sign turned 35.28% into "0.3528%" — a hundredfold understatement
        # that reads as a normal small number, which is exactly why it survived review.
        lead += f", {share * 100:.1f}% of the total"
    lead += f", from {top.get('row_count')} rows."

    return (
        f"{lead} {len(groups)} group(s) were compared{scale}.\n\n"
        f"Note: this is M4's fixed analysis — it always compares the first usable "
        f"numeric column against the first usable categorical column, and it has not "
        f"read your question ({question!r}). The reasoning arrives in M5."
    )


def _describe_only(
    question: str,
    dataset_id: uuid.UUID,
    version: int,
    schema: ToolResult,
    steps: list[dict[str, Any]],
    row_count: int | None,
) -> dict[str, Any]:
    """The honest result for a dataset the fixed analysis cannot compare.

    Reporting what was actually established beats inventing a comparison out of
    unsuitable columns — the failure mode this whole project exists to avoid.
    """
    kinds: dict[str, int] = {}
    for column in schema.data["columns"]:
        kinds[column.get("kind", "unknown")] = kinds.get(column.get("kind", "unknown"), 0) + 1
    breakdown = ", ".join(f"{count} {kind}" for kind, count in sorted(kinds.items()))

    return {
        "engine": FIXED_ENGINE,
        "question": question,
        "dataset": {"id": str(dataset_id), "version": version},
        "answer": (
            f"This dataset has {schema.data['column_count']} columns ({breakdown})"
            + (f" and {row_count:,} rows" if row_count else "")
            + ". M4's fixed analysis needs one groupable categorical column and one "
            "numeric column and could not find both, so it has not attempted a "
            f"comparison rather than forcing one. Your question ({question!r}) will be "
            "answered properly in M5."
        ),
        "steps": steps,
        "table": {
            "columns": ["name", "type", "kind"],
            "rows": [[c.get("name"), c.get("type"), c.get("kind")] for c in schema.data["columns"]],
        },
        "chart": None,
        "warnings": [],
    }
