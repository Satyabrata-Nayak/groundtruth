"""The agent loop, driven by a scripted model instead of a real one.

WHY A SCRIPTED MODEL AND NOT OLLAMA
-----------------------------------
Every behaviour worth testing here is the loop's, not the model's: does a tool error
come back as a repair message, is a repeated call refused, does an answer with no
evidence get pushed back, does the step budget stop the loop and still produce a
result. Running those against a real model would make each one slow, non-deterministic
and dependent on which weights happen to be pulled on the machine — three properties
that turn a test suite into decoration.

So `ScriptedModel` plays back a fixed list of turns. The tools underneath it are real,
the DuckDB queries are real, and the dataset is real: the only thing faked is the part
whose output we are not asserting on.

The genuinely non-deterministic question — "can qwen3:4b actually answer this?" — is
what `eval/` is for, and it is not a unit test.
"""

from __future__ import annotations

import uuid

import pytest

from app.agent.analyst import ENGINE, run_agent_analysis
from app.agent.contract import AnalysisFailed
from app.agent.evidence import chart_from_results, table_from_results
from app.agent.llm import ModelError, ModelTurn, ModelUnavailable, ToolCall, _parse_tool_calls
from app.agent.prompt import build_system_prompt, render_samples, render_schema
from app.data.service import create_dataset
from app.db.models import EventKind
from app.tools.base import ToolResult
from app.worker.heartbeat import StopRequested


class ScriptedModel:
    """An `LlmClient` that plays back prepared turns and records what it was sent.

    `available` and `error` exist so the two failure modes the loop distinguishes —
    "there is no model" and "the model call blew up mid-run" — can each be provoked
    without a network.
    """

    def __init__(
        self,
        turns,
        *,
        available: bool = True,
        error: Exception | None = None,
        exhausted: str = "I have run out of script.",
    ):
        self.turns = list(turns)
        self.exhausted = exhausted
        self.available = available
        self.error = error
        self.sent: list[list[dict]] = []
        self.tool_specs_seen: list[list[dict] | None] = []
        self.closed = False

    def check_available(self) -> None:
        if not self.available:
            raise ModelUnavailable("cannot reach the language model at http://test")

    def chat(self, messages, *, tools=None):
        if self.error is not None:
            raise self.error
        # Copied, because the loop keeps appending to the same list and a test that
        # asserts on what turn 1 was sent must not see turn 4's additions.
        self.sent.append([dict(m) for m in messages])
        self.tool_specs_seen.append(tools)
        if not self.turns:
            return ModelTurn(content=self.exhausted)
        return self.turns.pop(0)

    def close(self) -> None:
        self.closed = True


def says(text: str) -> ModelTurn:
    return ModelTurn(content=text)


def calls(name: str, **arguments) -> ModelTurn:
    raw = {"function": {"name": name, "arguments": arguments}}
    return ModelTurn(tool_calls=[ToolCall(name=name, arguments=arguments, raw=raw)])


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


@pytest.fixture
def run(sales_dataset):
    """Run the agent against the real dataset with a scripted model."""

    def go(turns, *, question="which region sells most?", checkpoint=None, **kwargs):
        events: list[tuple[EventKind, str]] = []
        model = ScriptedModel(turns, **kwargs)
        result = run_agent_analysis(
            dataset_id=sales_dataset.dataset_id,
            version=1,
            question=question,
            emit=lambda kind, message, payload=None: events.append((kind, message)),
            checkpoint=checkpoint or (lambda: None),
            client=model,
        )
        return result, events, model

    return go


# ============================================================== the happy path


def test_a_query_then_an_answer_produces_the_shared_result_shape(run, sales_dataset):
    """The contract both engines fill. Changing it changes the database column, the API
    schema and the frontend, so it is pinned here rather than assumed."""
    result, _, _ = run(
        [
            calls(
                "execute_sql",
                sql=(
                    "SELECT region, sum(revenue) AS total FROM dataset "
                    "GROUP BY region ORDER BY total DESC"
                ),
            ),
            says("North sells the most."),
        ]
    )

    assert set(result) == {
        "engine",
        "question",
        "dataset",
        "answer",
        "steps",
        "table",
        "chart",
        "warnings",
    }
    assert result["engine"] == ENGINE
    assert result["answer"] == "North sells the most."
    assert result["warnings"] == []
    assert result["dataset"] == {"id": str(sales_dataset.dataset_id), "version": 1}


def test_the_schema_is_fetched_before_the_model_is_asked_anything(run):
    """The agent's first tool call is not a decision — it is done for it.

    Letting the model discover the schema costs a full turn to produce a result that is
    deterministic, and it is the turn most likely to guess a column name.
    """
    result, events, model = run([says("done")])

    assert result["steps"][0]["tool"] == "inspect_schema"
    system = model.sent[0][0]["content"]
    assert "region" in system and "revenue" in system
    assert events[0][0] == EventKind.TOOL_CALL


def test_sample_rows_reach_the_prompt(run):
    """Three real rows are what tell the model that a value is 'West' and not 'west'."""
    _, _, model = run([says("done")])
    system = model.sent[0][0]["content"]
    assert "SAMPLE ROWS" in system
    assert "row 1:" in system


def test_the_evidence_table_comes_from_the_tool_result_not_the_model(run):
    """The answer is a claim; the table is what makes it checkable. If the model's
    prose and the table could disagree, only one of them was computed."""
    result, _, _ = run(
        [
            calls("compare_groups", group_column="region", metric_column="revenue"),
            says("West."),
        ]
    )
    assert result["table"]["columns"] == [
        "region",
        "sum(revenue)",
        "row_count",
        "share_of_total",
    ]
    assert {row[0] for row in result["table"]["rows"]} == {"North", "East", "West"}


def test_the_chart_type_is_chosen_from_the_shape_of_the_result(run):
    """Three groups with a share of the total are parts of a whole, so they are drawn as
    one. The type is inferred, not asked of the model: asking would cost a model turn,
    and the shape is fully known here."""
    result, _, _ = run(
        [
            calls("compare_groups", group_column="region", metric_column="revenue"),
            says("North."),
        ]
    )
    chart = result["chart"]["chart"]
    assert chart["type"] == "pie"
    assert len(chart["data"]) == 3
    assert chart["derived_from"] == "the final query result"


def test_a_long_ranking_is_drawn_as_bars(run):
    """A line between category names would draw a trend across things that have no
    order, which is a lie about the data rather than a busy chart."""
    result, _, _ = run(
        [
            calls(
                "execute_sql",
                sql="SELECT category, sum(revenue) AS total FROM dataset GROUP BY category",
            ),
            says("Books lead."),
        ]
    )
    assert result["chart"]["chart"]["type"] == "bar"


def test_a_single_row_answer_gets_no_chart(run):
    """A total has nothing to compare against. A bar chart of one bar is decoration
    presented as analysis, which is worse than no chart."""
    result, _, _ = run(
        [
            calls("execute_sql", sql="SELECT sum(revenue) AS total FROM dataset"),
            says("The total is 8,655."),
        ]
    )
    assert result["chart"] is None
    assert result["table"]["rows"] == [[8655.0]]


def test_the_model_call_is_recorded_as_an_event_without_its_reasoning(run):
    """MODEL_CALL is a thing that happened. What the model told itself on the way is
    narration, and storing it would invite the UI to present it as evidence."""
    _, events, _ = run(
        [calls("compare_groups", group_column="region", metric_column="revenue"), says("North.")]
    )
    model_events = [message for kind, message in events if kind == EventKind.MODEL_CALL]
    assert model_events == [
        "planning (round 1): compare_groups (0.0s)",
        "writing the answer (0.0s)",
    ]


# ============================================================== repair and refusal


def test_a_tool_error_is_handed_back_so_the_model_can_fix_it(run):
    """The reason `Registry.call` never raises. An exception would abort the run; a
    result with ok=False is a repair prompt naming the valid columns."""
    result, _, model = run(
        [
            calls("compare_groups", group_column="reveune", metric_column="revenue"),
            calls("compare_groups", group_column="region", metric_column="revenue"),
            says("North, and clearly."),
        ]
    )

    tool_messages = [m for turn in model.sent for m in turn if m["role"] == "tool"]
    assert any("reveune" in m["content"] and "region" in m["content"] for m in tool_messages)
    assert result["answer"] == "North, and clearly."
    assert [s["ok"] for s in result["steps"]] == [True, False, True]


def test_repeating_a_failed_call_is_refused_rather_than_re_run(run):
    """A repair round exists so the model can FIX a bad call, not repeat it. Re-running
    it produces the identical error and burns the round."""
    result, _, model = run(
        [
            calls("compare_groups", group_column="reveune", metric_column="revenue"),
            calls("compare_groups", group_column="reveune", metric_column="revenue"),
            says("I could not find that column."),
        ]
    )

    assert [s["tool"] for s in result["steps"]] == ["inspect_schema", "compare_groups"]
    refusals = [
        m
        for turn in model.sent
        for m in turn
        if m["role"] == "tool" and "you already called" in m["content"]
    ]
    assert refusals


def test_a_nameless_tool_call_gets_an_instruction_not_a_crash(run):
    result, _, model = run(
        [
            ModelTurn(tool_calls=[ToolCall(name="", arguments={}, raw={"function": {}})]),
            calls("compare_groups", group_column="region", metric_column="revenue"),
            says("done"),
        ]
    )
    tool_messages = [m for turn in model.sent for m in turn if m["role"] == "tool"]
    assert any("had no name" in m["content"] for m in tool_messages)
    assert result["answer"] == "done"


# ============================================================== grounding


def test_an_answer_with_no_query_behind_it_is_pushed_back_once(run):
    result, _, model = run(
        [
            says("Obviously the West region."),
            calls("compare_groups", group_column="region", metric_column="revenue"),
            says("North, and clearly."),
        ]
    )
    nudges = [
        m
        for turn in model.sent
        for m in turn
        if m["role"] == "user" and "without running anything" in m["content"]
    ]
    assert nudges
    assert result["answer"] == "North, and clearly."
    assert result["warnings"] == []


def test_an_ungrounded_answer_that_survives_the_push_back_is_flagged(run):
    """Kept, not discarded — but never presented as though it had been computed."""
    result, _, _ = run([says("Obviously the West."), says("Still obviously the West.")])
    assert result["answer"] == "Still obviously the West."
    assert result["warnings"] == ["the answer was written without running a query against the data"]


def test_an_empty_turn_asks_for_a_tool_call_or_an_answer(run):
    result, _, model = run(
        [
            ModelTurn(content=""),
            calls("compare_groups", group_column="region", metric_column="revenue"),
            says("done"),
        ]
    )
    prompts = [
        m
        for turn in model.sent
        for m in turn
        if m["role"] == "user" and "neither a tool call nor an answer" in m["content"]
    ]
    assert prompts
    assert result["answer"] == "done"


def test_an_inline_reasoning_block_is_stripped_from_the_answer(run):
    result, _, _ = run(
        [
            calls("compare_groups", group_column="region", metric_column="revenue"),
            says("<think>the user wants the top region</think>North, and clearly."),
        ]
    )
    assert result["answer"] == "North, and clearly."


# ============================================================== budgets


def test_the_answer_is_always_written_by_a_call_with_no_tools(run):
    """The single biggest speed decision in the agent, pinned so it cannot be undone.

    Measured on an identical conversation: with tools attached the model spends 41.8 s
    and 7,972 characters of reasoning re-deciding whether to call something else; with
    them omitted, 9.0 s and 1,547. Asking a model holding tools to stop using them is
    asking it to resist the strongest signal in its prompt.
    """
    _, _, model = run(
        [
            calls("compare_groups", group_column="region", metric_column="revenue"),
            says("North leads."),
        ]
    )
    assert model.tool_specs_seen[0] is not None  # planning gets the tools
    assert model.tool_specs_seen[-1] is None  # writing never does


def test_a_successful_round_ends_the_planning_phase_immediately(run):
    """Handing the model the tools again just to let it say "I am done" costs a full
    turn AND makes that turn four times slower. Once there is something to answer from,
    the tools go away."""
    _, _, model = run(
        [
            calls("compare_groups", group_column="region", metric_column="revenue"),
            says("North leads."),
        ]
    )
    assert len(model.tool_specs_seen) == 2


def test_a_failed_round_earns_a_repair_round(run):
    """The one case that needs the tools twice: the error names the valid columns."""
    result, _, model = run(
        [
            calls("compare_groups", group_column="reveune", metric_column="revenue"),
            calls("compare_groups", group_column="region", metric_column="revenue"),
            says("North leads."),
        ]
    )
    assert [s["ok"] for s in result["steps"]] == [True, False, True]
    assert model.tool_specs_seen[:2] == [model.tool_specs_seen[0], model.tool_specs_seen[0]]
    assert model.tool_specs_seen[-1] is None


def test_running_out_of_time_stops_before_the_next_model_call(run, monkeypatch):
    from app.config import get_settings

    monkeypatch.setattr(get_settings(), "agent_time_budget_s", 0.0)
    result, _, model = run([says("never reached")])

    assert len(model.sent) == 1  # only the forced final-answer call
    assert "time budget ran out" in result["warnings"][0]


def test_a_model_that_writes_nothing_still_returns_the_computed_result(run):
    """Losing real computation because the summarising call came back empty would be
    the worst trade available."""
    result, _, _ = run(
        [calls("compare_groups", group_column="region", metric_column="revenue")],
        exhausted="",
    )
    assert result["table"]["rows"]
    assert "did not write an answer" in result["answer"]
    assert "the model did not produce a written answer" in result["warnings"]


# ============================================================== failure modes


def test_no_model_running_fails_with_an_instruction_for_the_operator(run):
    with pytest.raises(AnalysisFailed, match="cannot reach the language model"):
        run([says("never asked")], available=False)


def test_a_model_error_mid_run_fails_the_analysis(run):
    with pytest.raises(AnalysisFailed, match="the language model failed"):
        run([], error=ModelError("connection reset"))


def test_an_unreadable_dataset_fails_before_the_model_is_consulted(db, data_root):
    model = ScriptedModel([says("never asked")])
    with pytest.raises(AnalysisFailed, match="could not read the dataset schema"):
        run_agent_analysis(
            dataset_id=uuid.uuid4(),
            version=1,
            question="q",
            emit=lambda *a: None,
            checkpoint=lambda: None,
            client=model,
        )
    assert not model.sent


def test_a_checkpoint_that_raises_stops_the_agent(run):
    """Cancellation takes effect between steps, which is the only place it can: the
    worker cannot interrupt a model call already in flight."""
    state = {"n": 0}

    def checkpoint():
        state["n"] += 1
        if state["n"] > 1:
            raise StopRequested("cancelled")

    with pytest.raises(StopRequested):
        run([calls("execute_sql", sql="SELECT 1 AS a"), says("x")], checkpoint=checkpoint)


# ============================================================== the pieces


def test_tool_arguments_arriving_as_a_json_string_are_parsed():
    """Ollama returns arguments as an object for most models and as a string for some.
    Fixed at the boundary, or it surfaces three layers away as an AttributeError."""
    calls_out = _parse_tool_calls(
        [{"function": {"name": "execute_sql", "arguments": '{"sql": "SELECT 1"}'}}]
    )
    assert calls_out[0].arguments == {"sql": "SELECT 1"}


def test_unparseable_tool_arguments_become_an_empty_dict_not_an_exception():
    calls_out = _parse_tool_calls([{"function": {"name": "execute_sql", "arguments": "{oh no"}}])
    assert calls_out[0].arguments == {}


def test_the_schema_renders_compactly_with_its_warnings():
    text = render_schema(
        {
            "row_count": 541909,
            "columns": [
                {"name": "Country", "type": "VARCHAR", "kind": "categorical", "distinct_count": 38},
                {
                    "name": "CustomerID",
                    "type": "BIGINT",
                    "kind": "numeric",
                    "distinct_count": 4372,
                    "null_fraction": 0.2493,
                    "warning": "high cardinality: behaves like an identifier",
                },
            ],
        }
    )
    assert "541,909 rows" in text
    assert "25% null" in text
    assert "[high cardinality]" in text


def test_a_long_value_in_a_sample_row_is_cut_rather_than_flooding_the_prompt():
    text = render_samples({"columns": ["note"], "rows": [["x" * 500]]})
    assert len(text) < 200


def test_the_system_prompt_carries_the_rules_that_matter():
    text = build_system_prompt({"row_count": 3, "columns": []}, None)
    assert "must come from a tool result" in text
    assert "execute_sql" in text
    assert "currency symbol" in text


def test_the_evidence_table_is_the_last_result_with_rows_not_the_first():
    """An agent's early calls are orientation. The call that produced the answer is the
    last one that returned rows; picking the first shows a person its throat-clearing."""
    first = ToolResult(
        tool="execute_sql", ok=True, data={"columns": ["a"], "rows": [[1]]}, summary=""
    )
    last = ToolResult(
        tool="execute_sql", ok=True, data={"columns": ["b"], "rows": [[2]]}, summary=""
    )
    assert table_from_results([first, last])["columns"] == ["b"]


def test_a_chart_the_agent_asked_for_beats_one_derived_from_the_table():
    """`create_chart` was a deliberate choice about presentation. The fallback is not."""
    made = ToolResult(tool="create_chart", ok=True, data={"chart": {"type": "line"}}, summary="")
    table = {"columns": ["x", "y"], "rows": [["a", 1], ["b", 2]]}
    assert chart_from_results([made], table, "q")["chart"]["type"] == "line"


def test_a_failed_result_is_never_used_as_evidence():
    broken = ToolResult(tool="execute_sql", ok=False, error="boom", data={})
    assert table_from_results([broken]) is None


# ============================================================== answer verification


def result_with(**data):
    return ToolResult(tool="execute_sql", ok=True, data=data, summary="")


def test_the_arithmetic_the_model_did_in_its_head_is_caught():
    """The real failure, quoted from the first end-to-end run on the retail data.

    Both product totals came from DuckDB. The GAP did not: 53,847 - 47,363 is 6,484,
    and the model wrote 16,484 in a sentence where everything else was correct.
    """
    from app.agent.verify import untraceable_figures

    results = [
        result_with(
            columns=["Description", "TotalUnits"],
            rows=[["WORLD WAR 2 GLIDERS", 53847], ["JUMBO BAG RED RETROSPOT", 47363]],
        )
    ]
    answer = (
        "WORLD WAR 2 GLIDERS sold the most units at 53,847, exceeding "
        "JUMBO BAG RED RETROSPOT by 16,484 units."
    )
    assert untraceable_figures(answer, results) == ["16,484"]


def test_a_figure_written_with_separators_still_matches_the_computed_one():
    from app.agent.verify import untraceable_figures

    results = [result_with(columns=["total"], rows=[[8187806.363998184]])]
    assert untraceable_figures("The total is 8,187,806.36.", results) == []


def test_a_fraction_written_as_a_percentage_matches():
    """`share_of_total` is a fraction and every answer writes it as a percent. A checker
    that did not know this would flag every correctly written share."""
    from app.agent.verify import untraceable_figures

    results = [result_with(groups=[{"group": "West", "share_of_total": 0.3528}])]
    assert untraceable_figures("West accounts for 35.28% of the total.", results) == []


def test_small_numbers_are_not_flagged():
    """ "the top 10 countries" and "3 groups" are numbers no tool needs to have produced.
    Warning about them would bury the one figure that matters."""
    from app.agent.verify import untraceable_figures

    results = [result_with(columns=["x"], rows=[[999999]])]
    assert untraceable_figures("Across the top 10 of 3 groups, 999999 leads.", results) == []


def test_a_number_from_a_failed_tool_result_is_not_evidence():
    from app.agent.verify import untraceable_figures

    failed = ToolResult(tool="execute_sql", ok=False, error="boom", data={"rows": [[123456]]})
    good = result_with(columns=["x"], rows=[[1]])
    assert untraceable_figures("The answer is 123456.", [failed, good]) == ["123456"]


def test_the_warning_names_the_untraceable_figures(run):
    """Attached to the result, not used to rewrite the sentence. A number the system
    cannot trace is a fact about its confidence, and hiding it asserts more than is
    known."""
    result, _, _ = run(
        [
            calls("compare_groups", group_column="region", metric_column="revenue"),
            says("North leads with a margin of 91,827 over the others."),
        ]
    )
    assert len(result["warnings"]) == 1
    assert "91,827" in result["warnings"][0]
    assert "calculated by the model" in result["warnings"][0]


def test_a_fully_grounded_answer_gets_no_warning(run):
    result, _, _ = run(
        [
            calls("compare_groups", group_column="region", metric_column="revenue"),
            says("North leads, and the three regions are close."),
        ]
    )
    assert result["warnings"] == []


def test_looking_at_the_schema_counts_as_evidence_for_a_refusal(run):
    """The real false positive: asked for customer ages on a dataset with none, the
    agent inspected the schema, correctly said the data cannot answer it, and was told
    it had "written an answer without running a query". Establishing that the data
    cannot answer a question IS done by looking at the data."""
    result, _, _ = run(
        [
            calls("inspect_schema", include_statistics=False),
            says("This dataset records no customer age, so the average cannot be computed."),
        ],
        question="what is the average age of our customers?",
    )
    assert result["warnings"] == []
