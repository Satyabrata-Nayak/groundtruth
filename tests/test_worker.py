"""The worker: column choice, the answer it writes, and the four outcomes it handles.

THE FIRST HALF OF THIS FILE IS A BUG MUSEUM
-------------------------------------------
Four defects reached a working end-to-end run before anything caught them, and every
one produced output that was *correct arithmetic about the wrong thing* — the failure
mode this whole project exists to prevent. None was visible by reading the code; all
four were obvious the moment the output was read. They get named tests so they cannot
come back quietly.

    picked `order_id` as the metric        "total order_id by region = 4,410,927"
    rejected every float as an identifier  silently downgraded to a worse column
    read the group label by column name    "region = None" above a table saying West
    printed a fraction with a % sign       35.28% rendered as "0.3528%"
"""

from __future__ import annotations

import uuid

import pytest

from app.data.service import create_dataset
from app.db.models import Analysis, AnalysisStatus, EventKind
from app.db.session import session_scope
from app.jobs import queue
from app.tools.base import ToolResult
from app.worker.analysis import (
    AnalysisFailed,
    _pick_group_column,
    _pick_metric_column,
    _write_answer,
    run_analysis,
)
from app.worker.heartbeat import StopRequested
from app.worker.loop import Worker, make_worker_id


def column(name, kind, type_="DOUBLE", distinct=None, warning=None):
    entry = {"name": name, "kind": kind, "type": type_, "distinct_count": distinct}
    if warning:
        entry["warning"] = warning
    return entry


# ================================================================ column choice


def test_an_integer_column_with_one_value_per_row_is_an_identifier():
    """Bug 1. `inspect_schema` does not warn about this: `is_high_cardinality` is only
    computed for CATEGORICAL columns, so a numeric primary key arrives unflagged."""
    columns = [
        column("order_id", "numeric", "BIGINT", distinct=5000),
        column("revenue", "numeric", "DOUBLE", distinct=4850),
    ]
    assert _pick_metric_column(columns, row_count=5000) == "revenue"


def test_a_float_with_one_value_per_row_is_a_measurement_not_an_identifier():
    """Bug 2, which the fix for bug 1 introduced.

    A continuous measurement over 5,000 rows also has ~5,000 distinct values. Judging
    on the distinct ratio alone rejected `revenue`, `cost` and `unit_price` and left
    the analysis summing whichever small-range integer happened to survive.
    """
    columns = [column("unit_price", "numeric", "DOUBLE", distinct=4998)]
    assert _pick_metric_column(columns, row_count=5000) == "unit_price"


def test_an_id_named_column_is_used_only_when_nothing_else_qualifies():
    """The name rule is a preference, never a veto: `bid`, `grid` and `void` all end
    in 'id', and a real identifier may be called `customer_number`."""
    only_ids = [column("account_id", "numeric", "BIGINT", distinct=3)]
    assert _pick_metric_column(only_ids, row_count=1000) == "account_id"

    with_alternative = [
        column("account_id", "numeric", "BIGINT", distinct=3),
        column("spend", "numeric", "DOUBLE", distinct=900),
    ]
    assert _pick_metric_column(with_alternative, row_count=1000) == "spend"


def test_a_small_table_is_never_judged_on_cardinality():
    """With eight rows, "eight distinct values" is not evidence of anything."""
    columns = [column("units", "numeric", "BIGINT", distinct=8)]
    assert _pick_metric_column(columns, row_count=8) == "units"


def test_constant_and_high_cardinality_columns_are_skipped():
    columns = [
        column("currency", "categorical", "VARCHAR", 1, warning="constant"),
        column("email", "categorical", "VARCHAR", 990, warning="high cardinality"),
        column("region", "categorical", "VARCHAR", 4),
    ]
    assert _pick_group_column(columns) == "region"


def test_a_grouping_column_with_too_many_groups_is_skipped():
    """A 200-bar chart is not a chart."""
    columns = [column("city", "categorical", "VARCHAR", 200)]
    assert _pick_group_column(columns) is None


def test_no_suitable_columns_returns_none():
    assert _pick_group_column([column("note", "categorical", "VARCHAR", 1)]) is None
    assert _pick_metric_column([column("name", "categorical", "VARCHAR", 5)], 100) is None


# ================================================================== the answer


def comparison(groups):
    return ToolResult(tool="compare_groups", ok=True, data={"groups": groups})


def test_the_answer_reads_the_group_label_from_the_right_key():
    """Bug 3. `compare_groups` keys the label as "group", not as the column's name.
    Reading `top[group_column]` gave None — printed directly above a table that said
    otherwise, which is worse than an error because it looks like an answer."""
    result = comparison([{"group": "West", "value": 150.75, "row_count": 2}])
    answer = _write_answer("q", "region", "revenue", result, 6)
    assert "'West'" in answer
    assert "None" not in answer


def test_share_of_total_is_rendered_as_a_percentage():
    """Bug 4. The tool returns value/total — a fraction. Appending '%' to 0.3528
    understates it a hundredfold, and 0.35% is a plausible-looking number, which is
    exactly why nobody notices."""
    result = comparison(
        [{"group": "West", "value": 150.75, "row_count": 2, "share_of_total": 0.3528}]
    )
    answer = _write_answer("q", "region", "revenue", result, 6)
    assert "35.3% of the total" in answer
    assert "0.3528%" not in answer


def test_the_answer_admits_it_did_not_read_the_question():
    """M4 does not answer the question, and the UI must not imply that it did."""
    result = comparison([{"group": "West", "value": 1, "row_count": 1}])
    answer = _write_answer("why did profit fall?", "region", "revenue", result, 10)
    assert "why did profit fall?" in answer
    assert "M5" in answer


def test_an_empty_comparison_says_so_rather_than_indexing_into_nothing():
    answer = _write_answer("q", "region", "revenue", comparison([]), 0)
    assert "nothing" in answer.lower()


# ============================================================ the whole analysis


@pytest.fixture
def sales_dataset(db, data_root, tmp_path):
    path = tmp_path / "sales.csv"
    path.write_text(
        "order_id,region,category,revenue,units\n"
        + "".join(
            f"{i},{'West' if i % 3 == 0 else 'East' if i % 3 == 1 else 'North'},"
            f"{'Books' if i % 2 else 'Toys'},{100 + i * 1.5},{1 + i % 4}\n"
            for i in range(60)
        ),
        encoding="utf-8",
    )
    return create_dataset(path, name="sales")


def collect(events):
    def emit(kind, message, payload=None):
        events.append((kind, message))

    return emit


def test_run_analysis_produces_the_m5_result_shape(sales_dataset):
    """The contract M5 has to fill. If this shape changes, so do the database column,
    the API schema and the frontend — so it is pinned here rather than assumed."""
    events = []
    result = run_analysis(
        dataset_id=sales_dataset.dataset_id,
        version=1,
        question="which region sells most?",
        emit=collect(events),
        checkpoint=lambda: None,
    )

    assert set(result) == {"engine", "question", "dataset", "answer", "steps", "table", "chart"}
    assert result["engine"] == "hardcoded-v1"
    assert result["question"] == "which region sells most?"
    assert result["dataset"] == {"id": str(sales_dataset.dataset_id), "version": 1}

    assert [step["tool"] for step in result["steps"]] == [
        "inspect_schema",
        "compare_groups",
        "create_chart",
    ]
    assert all(step["ok"] for step in result["steps"])
    assert result["table"]["columns"][0] == "group"
    assert result["chart"] is not None


def test_it_does_not_sum_the_identifier(sales_dataset):
    """The end-to-end version of the first bug, on data with a real numeric id."""
    result = run_analysis(
        dataset_id=sales_dataset.dataset_id,
        version=1,
        question="q",
        emit=collect([]),
        checkpoint=lambda: None,
    )
    metric = result["steps"][1]["arguments"]["metric_column"]
    assert metric == "revenue"
    assert "order_id" not in result["answer"]


def test_every_tool_call_is_announced_before_and_after(sales_dataset):
    """The trail must show intent as well as outcome. In M5 a TOOL_CALL with no
    matching TOOL_RESULT is how a hung or killed step is recognised."""
    events = []
    run_analysis(
        dataset_id=sales_dataset.dataset_id,
        version=1,
        question="q",
        emit=collect(events),
        checkpoint=lambda: None,
    )
    calls = [message for kind, message in events if kind == EventKind.TOOL_CALL]
    results = [message for kind, message in events if kind == EventKind.TOOL_RESULT]
    assert len(calls) == len(results) == 3


def test_a_checkpoint_that_raises_stops_the_analysis(sales_dataset):
    """Cancellation has to take effect between steps, not only at the end."""
    calls = {"n": 0}

    def checkpoint():
        calls["n"] += 1
        if calls["n"] > 1:
            raise StopRequested("cancelled")

    with pytest.raises(StopRequested):
        run_analysis(
            dataset_id=sales_dataset.dataset_id,
            version=1,
            question="q",
            emit=collect([]),
            checkpoint=checkpoint,
        )


def test_an_unreadable_dataset_fails_with_a_reason(db, data_root):
    """A dataset id that resolves to no files must produce AnalysisFailed — a message
    for the user — rather than an unhandled exception the worker reports as a bug."""
    with pytest.raises(AnalysisFailed):
        run_analysis(
            dataset_id=uuid.uuid4(),
            version=1,
            question="q",
            emit=collect([]),
            checkpoint=lambda: None,
        )


def test_a_dataset_with_nothing_to_compare_is_described_instead(db, data_root, tmp_path):
    """Refusing to force a comparison out of unsuitable columns is the point.

    Every column here is free text, so there is no metric to sum. Inventing one would
    be the exact failure this project exists to avoid.
    """
    path = tmp_path / "notes.csv"
    path.write_text(
        "title,body\n" + "".join(f"note {i},some text {i}\n" for i in range(30)),
        encoding="utf-8",
    )
    created = create_dataset(path, name="notes")

    result = run_analysis(
        dataset_id=created.dataset_id,
        version=1,
        question="what is in here?",
        emit=collect([]),
        checkpoint=lambda: None,
    )
    assert result["chart"] is None
    assert "could not find both" in result["answer"]
    assert [step["tool"] for step in result["steps"]] == ["inspect_schema"]


# ================================================================== the loop


@pytest.mark.integration
def test_the_worker_claims_runs_and_completes(sales_dataset):
    with session_scope() as session:
        analysis, _ = queue.enqueue(
            session,
            dataset_id=sales_dataset.dataset_id,
            dataset_version=1,
            question="which region sells most?",
        )
        analysis_id = analysis.id

    worker = Worker(worker_id="test-worker")
    assert worker.tick() is True
    assert worker.tick() is False  # queue is empty again

    with session_scope() as session:
        stored = session.get(Analysis, analysis_id)
        assert stored.status == AnalysisStatus.SUCCEEDED
        assert stored.result["engine"] == "hardcoded-v1"
        assert stored.finished_at is not None
        assert stored.error is None


@pytest.mark.integration
def test_a_failing_analysis_becomes_a_failed_row_not_a_stuck_one(db, data_root, monkeypatch):
    """A crash inside the analysis must reach a terminal state. Left RUNNING, it would
    sit there until the sweep guessed, and the user would watch a spinner for 30
    seconds to be told nothing useful."""
    from app.worker import loop as loop_module

    def explode(**_kwargs):
        raise RuntimeError("duckdb went to lunch")

    monkeypatch.setattr(loop_module, "run_analysis", explode)

    from app.db.models import Dataset

    with session_scope() as session:
        dataset = Dataset(id=uuid.uuid4(), name="doomed")
        session.add(dataset)
        session.flush()
        analysis, _ = queue.enqueue(session, dataset_id=dataset.id, dataset_version=1, question="q")
        analysis_id = analysis.id

    assert Worker(worker_id="test-worker").tick() is True

    with session_scope() as session:
        stored = session.get(Analysis, analysis_id)
        assert stored.status == AnalysisStatus.FAILED
        # The type name is in the message: "internal error: RuntimeError: ..." tells a
        # maintainer this was a bug in us, not a bad question.
        assert "RuntimeError" in stored.error
        assert "duckdb went to lunch" in stored.error


@pytest.mark.integration
def test_a_cancelled_analysis_is_recorded_as_cancelled_not_failed(db, data_root, monkeypatch):
    from app.worker import loop as loop_module

    def stopped(**_kwargs):
        raise StopRequested("cancelled")

    monkeypatch.setattr(loop_module, "run_analysis", stopped)

    from app.db.models import Dataset

    with session_scope() as session:
        dataset = Dataset(id=uuid.uuid4(), name="cancelled")
        session.add(dataset)
        session.flush()
        analysis, _ = queue.enqueue(session, dataset_id=dataset.id, dataset_version=1, question="q")
        analysis_id = analysis.id

    Worker(worker_id="test-worker").tick()

    with session_scope() as session:
        assert session.get(Analysis, analysis_id).status == AnalysisStatus.CANCELLED


@pytest.mark.integration
def test_a_worker_that_lost_its_job_writes_nothing(db, data_root, monkeypatch):
    """The counter-intuitive outcome, and the one that is always forgotten.

    Losing ownership means another worker is now responsible for this analysis. The
    correct behaviour is to drop a perfectly good result on the floor, because writing
    it would be a second, conflicting story about the same row.
    """
    from app.db.models import Dataset
    from app.worker import loop as loop_module

    def lost(**_kwargs):
        raise StopRequested("lost")

    monkeypatch.setattr(loop_module, "run_analysis", lost)

    with session_scope() as session:
        dataset = Dataset(id=uuid.uuid4(), name="lost")
        session.add(dataset)
        session.flush()
        analysis, _ = queue.enqueue(session, dataset_id=dataset.id, dataset_version=1, question="q")
        analysis_id = analysis.id

    Worker(worker_id="test-worker").tick()

    with session_scope() as session:
        stored = session.get(Analysis, analysis_id)
        # Still RUNNING: this worker deliberately said nothing, and the sweep will
        # deal with it.
        assert stored.status == AnalysisStatus.RUNNING
        assert stored.result is None
        assert stored.error is None


def test_worker_ids_are_unique_per_process():
    """Host and pid alone are not enough: a restarted worker can be given the same pid,
    and a stale row naming it would then pass an ownership guard it should fail."""
    assert make_worker_id() != make_worker_id()
