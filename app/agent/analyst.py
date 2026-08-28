"""The agent loop: a question in, a checked answer out.

    question ──► system prompt (rules + real schema + real sample rows)
                     │
                     ▼
              ┌──► model turn ──┬── tool calls ──► ToolRegistry.call ──► DuckDB
              │                 │                        │
              │                 │        result JSON ◄───┘
              └─────────────────┤        (fed back as a tool message)
                                │
                                └── prose, no tool calls ──► the answer
                     │
                     ▼
              evidence table + chart, built from the tool results, not from the model

WHAT THIS LOOP IS ACTUALLY DEFENDING AGAINST
--------------------------------------------
A 4B model is entirely capable of writing a fluent, specific, wrong answer. Every
guard below exists because that is the default outcome, not an edge case:

    the schema is in the prompt        so column names are read, not invented
    tools are the only computation     so numbers come from DuckDB, not from the model
    errors are fed back as text        so a bad call is repaired instead of retried
    repeat calls are refused           so the loop cannot spin on one idea
    a step budget and a time budget    so a confused agent stops and says what it has
    an answer with no tool call        is pushed back once, and flagged if it persists
    the table comes from the results   so the evidence cannot agree with a wrong answer

THE BUDGETS ARE TWO, NOT ONE
----------------------------
Steps bound how many decisions the model makes; wall-clock bounds how long a person
waits. They are different failure modes. Six fast steps that go nowhere should stop at
six; one step that takes four minutes because the model is swapping should stop on
time. Hitting either one is not an error — the loop asks for a final answer with the
tools removed, and the model reports what it established.

WHY THE FINAL TURN DROPS THE TOOLS ENTIRELY
--------------------------------------------
Asking a model to "stop calling tools now" while still handing it tools is asking it to
resist the strongest signal in its prompt. Removing them from the request makes a tool
call impossible rather than discouraged.
"""

from __future__ import annotations

import json
import re
import time
import uuid
from typing import Any

from app.agent.contract import AnalysisFailed, Checkpoint, Emit
from app.agent.evidence import chart_from_results, table_from_results
from app.agent.llm import LlmClient, ModelError, ModelTurn, ModelUnavailable, ToolCall
from app.agent.prompt import (
    ANSWER_SYSTEM_PROMPT,
    EMPTY_TURN_PROMPT,
    FORCE_ANSWER_PROMPT,
    NO_EVIDENCE_PROMPT,
    build_system_prompt,
    build_user_prompt,
)
from app.agent.verify import verification_warning
from app.config import get_settings
from app.data.sandbox import TABLE_NAME
from app.db.models import EventKind
from app.tools import get_registry
from app.tools.base import ToolContext, ToolRegistry, ToolResult

# Written into every result this engine produces. A stored analysis says what made it.
ENGINE = "agent-v1"

# The action space offered to the model, in the order it is most likely to need them.
# Narrower than the full registry on purpose: `correlation` is a real tool that a 4B
# model reaches for when it sees two numeric columns and the question said nothing
# about a relationship, and every unused tool is a distractor in every prompt.
AGENT_TOOLS = ["inspect_schema", "profile_column", "execute_sql", "compare_groups", "create_chart"]

# At most this many tool calls are honoured from one model turn.
#
# Raised from 2 to 4 once the economics were measured. A tool call costs 30-70 ms and a
# model turn costs 45-90 seconds, so four queries in one round are indistinguishable in
# time from one — and they are the difference between "the top 10 products" and "the top
# products, the busiest countries, and the overall total to put them against".
#
# The cap still exists because qwen3 sometimes emits the same call three times in
# parallel. Duplicates within a turn are dropped before this cap applies.
_MAX_CALLS_PER_TURN = 4

# A tool result is a message in a context window. `max_tool_result_rows` already caps
# the rows; this caps the characters, for the case of 50 rows of long text.
_MAX_TOOL_PAYLOAD_CHARS = 8000

# Some models emit their reasoning inline despite Ollama returning it separately.
_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def run_agent_analysis(
    *,
    dataset_id: uuid.UUID,
    version: int,
    question: str,
    emit: Emit,
    checkpoint: Checkpoint,
    llm_model: str | None = None,
    llm_thinking: bool | None = None,
    history: str = "",
    client: LlmClient | None = None,
    registry: ToolRegistry | None = None,
) -> dict[str, Any]:
    """Answer `question` about one dataset version, and return the result payload.

    `client` and `registry` are injectable so the whole loop can be exercised against a
    scripted model in a unit test — the alternative is a test suite that only runs when
    Ollama happens to be up, which is a test suite that stops being run.
    """
    settings = get_settings()
    registry = registry or get_registry()
    context = ToolContext(dataset_id=dataset_id, version=version)

    owns_client = client is None
    # The asker's choice wins over configuration; configuration wins over the model's
    # own default. `llm_thinking` is tri-state — None means "nobody said", which is not
    # the same as False, and collapsing them would silently disable reasoning for every
    # request that did not mention it.
    client = client or LlmClient(model=llm_model, think=llm_thinking)
    try:
        return _run(
            client=client,
            registry=registry,
            context=context,
            dataset_id=dataset_id,
            version=version,
            question=question,
            emit=emit,
            checkpoint=checkpoint,
            history=history,
            max_steps=settings.agent_max_steps,
            max_tool_rounds=settings.agent_max_tool_rounds,
            time_budget_s=settings.agent_time_budget_s,
        )
    finally:
        if owns_client:
            client.close()


def _run(
    *,
    client: LlmClient,
    registry: ToolRegistry,
    context: ToolContext,
    dataset_id: uuid.UUID,
    version: int,
    question: str,
    emit: Emit,
    checkpoint: Checkpoint,
    history: str,
    max_steps: int,
    max_tool_rounds: int,
    time_budget_s: float,
) -> dict[str, Any]:
    deadline = time.monotonic() + time_budget_s
    steps: list[dict[str, Any]] = []
    results: list[ToolResult] = []
    warnings: list[str] = []

    # --- 0. everything deterministic, before a single token is generated -----------
    #
    # The schema is not something to ask a model for. Fetching it here costs
    # milliseconds, cannot be wrong, and removes the agent's most error-prone turn.
    checkpoint()
    schema = _call_tool(
        "inspect_schema", {"include_statistics": True}, registry, context, steps, results, emit
    )
    if not schema.ok:
        raise AnalysisFailed(f"could not read the dataset schema: {schema.error}")
    samples = _sample_rows(registry, context)

    # --- 1. is there a model at all? ----------------------------------------------
    #
    # Checked before the loop so "ollama is not running" fails in 200 ms with an
    # instruction, rather than after the first 30 second timeout with a stack trace.
    try:
        client.check_available()
    except ModelUnavailable as exc:
        raise AnalysisFailed(str(exc)) from exc

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": build_system_prompt(schema.data, samples, history)},
        {"role": "user", "content": build_user_prompt(question)},
    ]
    specs = registry.specs(only=AGENT_TOOLS)
    attempted: set[tuple[str, str]] = set()
    # Everything already in `results` was fetched to build the prompt, not chosen.
    chosen_from = len(results)
    nudged_for_evidence = False
    answer = ""
    calls_made = 0

    def has_evidence() -> bool:
        """Did anything the MODEL chose to run succeed?

        Not "anything but inspect_schema". Excluding that tool by name once flagged a
        correct refusal: asked for customer ages on a dataset that has none, the agent
        inspected the schema, said so, and was told it had "written an answer without
        running a query". Establishing that data cannot answer a question is done by
        looking at the data. The automatic pre-fetch is excluded because nobody chose it.
        """
        return any(r.ok for r in results[chosen_from:])

    # --- 2. planning: the only phase that carries the tools --------------------------
    #
    # Each round, the model may ask for several tools at once and they all run before it
    # sees any of them. That is deliberate: a tool costs 40 ms and a model turn costs 45
    # seconds, so three queries in one round are free next to three rounds of one.
    #
    # The loop leaves as soon as there is something to answer from. Handing the model
    # the tools again to let it say "I am done" costs a full turn AND makes that turn
    # four times slower, because a model looking at tools re-argues whether to use them.
    for round_number in range(1, max_tool_rounds + 1):
        checkpoint()

        if time.monotonic() >= deadline:
            warnings.append(f"the {time_budget_s:.0f}s time budget ran out while planning")
            break

        turn = _model_turn(client, messages, specs, emit, label=f"planning (round {round_number})")
        calls_made += 1

        if turn.wants_tools:
            messages.append(_assistant_message(turn))
            for call in _calls_to_run(turn.tool_calls):
                checkpoint()
                messages.append(
                    _run_one_call(call, registry, context, steps, results, emit, attempted)
                )
            # Something worked, so there is an answer to write. Another tools-attached
            # round would only re-litigate a decision already made.
            if has_evidence():
                break
            # Everything failed. THIS is what the second round is for: the errors name
            # the valid columns, and a repair usually lands.
            continue

        # No tool calls: the model believes it can answer already.
        answer = _clean(turn.content)
        if has_evidence() or nudged_for_evidence:
            if not has_evidence():
                warnings.append("the answer was written without running a query against the data")
            break

        # An answer with nothing behind it, or an empty turn. Each gets one correction,
        # and the correction differs. `nudged_for_evidence` is what makes it ONE: without
        # it the next ungrounded answer is pushed back again, and an agent that has
        # decided the data cannot answer the question is asked to query it forever.
        messages.append({"role": "assistant", "content": turn.content})
        if answer:
            nudged_for_evidence = True
            messages.append({"role": "user", "content": NO_EVIDENCE_PROMPT})
        else:
            messages.append({"role": "user", "content": EMPTY_TURN_PROMPT})
        answer = ""

    # --- 3. the answer, written by a call with NO TOOLS ATTACHED ---------------------
    #
    # This is the single biggest speed decision in the agent. The same conversation,
    # asked for the same prose:
    #
    #     tools attached      41.8 s   2,098 output tokens   7,972 chars of thinking
    #     tools omitted        9.0 s     431 output tokens   1,547 chars of thinking
    #
    # A model holding tools spends its reasoning deciding whether to use them again.
    # Take them away and it does the job it was asked to do.
    if not answer and calls_made < max_steps:
        answer = _final_answer(client, messages, emit)

    table = table_from_results(results)
    if not answer:
        # The model produced nothing usable. The computation still happened, so report
        # that honestly rather than failing and throwing away real work.
        warnings.append("the model did not produce a written answer")
        answer = (
            "The analysis ran but the model did not write an answer. The computed "
            "result is shown below; the numbers in it are exact."
        )

    # The last guard, and the only one that reads what the model actually WROTE. Every
    # other check constrains what it may do; this one checks what it said.
    untraceable = verification_warning(answer, results)
    if untraceable:
        warnings.append(untraceable)

    emit(
        EventKind.NOTE,
        "answer written" + (" (with unverified figures)" if untraceable else ""),
        {"characters": len(answer), "verified": untraceable is None},
    )

    return {
        "engine": ENGINE,
        # Which model actually answered. `engine` says "an agent did this"; this says
        # which one, and two answers to the same question can differ for no reason
        # other than this field.
        "model": client.model,
        "thinking": client.think,
        "question": question,
        "dataset": {"id": str(dataset_id), "version": version},
        "answer": answer,
        "steps": steps,
        "table": table,
        "chart": chart_from_results(results, table, question),
        "warnings": warnings,
    }


# --------------------------------------------------------------------------------
# One model turn
# --------------------------------------------------------------------------------


def _model_turn(
    client: LlmClient,
    messages: list[dict[str, Any]],
    specs: list[dict[str, Any]] | None,
    emit: Emit,
    *,
    label: str,
) -> ModelTurn:
    """One call to the model, recorded as an event.

    A `ModelUnavailable` mid-run is fatal: the server went away and no amount of
    retrying inside the loop starts it again. Any other `ModelError` is fatal too, but
    for a different reason — a timeout here has already consumed most of the budget,
    and a retry would spend the rest of it the same way.
    """
    try:
        turn = client.chat(messages, tools=specs)
    except ModelUnavailable as exc:
        raise AnalysisFailed(str(exc)) from exc
    except ModelError as exc:
        raise AnalysisFailed(f"the language model failed: {exc}") from exc

    intent = (
        ", ".join(call.name or "?" for call in turn.tool_calls)
        if turn.tool_calls
        else "ready to answer"
    )
    emit(
        EventKind.MODEL_CALL,
        f"{label}: {intent} ({turn.wall_s:.1f}s)",
        {
            "phase": label,
            "seconds": round(turn.wall_s, 2),
            "prompt_tokens": turn.prompt_tokens,
            "output_tokens": turn.output_tokens,
            # The reasoning text itself is deliberately NOT stored: it is narration,
            # not evidence, and a UI given it would show it as though it were.
            "thinking_characters": len(turn.thinking),
        },
    )
    return turn


def _final_answer(client: LlmClient, messages: list[dict[str, Any]], emit: Emit) -> str:
    """Ask for prose with the tools removed, so a tool call is impossible — and fast.

    Failures here are swallowed: the tool results are already in hand, and losing a
    completed analysis because the summarising call timed out would be the worst
    possible trade.
    """
    # The SYSTEM PROMPT IS SWAPPED, not just the tools. The planning prompt is a page of
    # rules about choosing tools, batching calls and repairing failed queries, none of
    # which applies once the results are in — and a small model does not ignore
    # instructions it cannot use, it reasons about them at length. Removing the tools
    # took the answer turn from 165 s to 56 s; removing the rules it no longer needs is
    # the rest of that same saving.
    conversation = [
        {"role": "system", "content": ANSWER_SYSTEM_PROMPT},
        *messages[1:],
        {"role": "user", "content": FORCE_ANSWER_PROMPT},
    ]
    try:
        turn = client.chat(conversation, tools=None)
    except ModelError:
        return ""
    emit(
        EventKind.MODEL_CALL,
        f"writing the answer ({turn.wall_s:.1f}s)",
        {
            "phase": "writing the answer",
            "seconds": round(turn.wall_s, 2),
            "prompt_tokens": turn.prompt_tokens,
            "output_tokens": turn.output_tokens,
            "thinking_characters": len(turn.thinking),
        },
    )
    return _clean(turn.content)


# --------------------------------------------------------------------------------
# Tool calls
# --------------------------------------------------------------------------------


def _calls_to_run(calls: list[ToolCall]) -> list[ToolCall]:
    """Drop duplicates within a turn and cap how many are honoured."""
    seen: set[tuple[str, str]] = set()
    chosen: list[ToolCall] = []
    for call in calls:
        key = _call_key(call.name, call.arguments)
        if key in seen:
            continue
        seen.add(key)
        chosen.append(call)
        if len(chosen) == _MAX_CALLS_PER_TURN:
            break
    return chosen


def _run_one_call(
    call: ToolCall,
    registry: ToolRegistry,
    context: ToolContext,
    steps: list[dict[str, Any]],
    results: list[ToolResult],
    emit: Emit,
    attempted: set[tuple[str, str]],
) -> dict[str, Any]:
    """Execute one proposed call and build the tool message that answers it.

    Every branch returns a `tool` message, including the refusals. A model that asked
    for something and got silence has no way to recover; a model that got a sentence
    explaining what to do instead usually recovers on the next turn.
    """
    if not call.name:
        return _tool_message(
            "unknown",
            {"ok": False, "error": "the tool call had no name. Name one of the tools provided."},
        )

    key = _call_key(call.name, call.arguments)
    if key in attempted:
        # Not an error and not free: re-running it would produce the identical payload
        # and consume a step. Say so, and the model moves on instead of spinning.
        return _tool_message(
            call.name,
            {
                "ok": False,
                "error": (
                    f"you already called {call.name} with exactly these arguments and its "
                    f"result is above. Use that result, or call something different."
                ),
            },
        )
    attempted.add(key)

    result = _call_tool(call.name, call.arguments, registry, context, steps, results, emit)
    return _tool_message(call.name, result.to_model_payload())


def _call_tool(
    name: str,
    arguments: dict[str, Any],
    registry: ToolRegistry,
    context: ToolContext,
    steps: list[dict[str, Any]],
    results: list[ToolResult],
    emit: Emit,
) -> ToolResult:
    """Announce a call, run it, and record the outcome everywhere it belongs.

    TOOL_CALL is emitted BEFORE execution and TOOL_RESULT after, always in that order
    and always both — including for the schema fetch the agent did not choose. A
    TOOL_CALL with no matching TOOL_RESULT is precisely how a hung or killed step is
    recognised in the trail, and an exception to that rule anywhere makes the rule
    useless everywhere.
    """
    emit(EventKind.TOOL_CALL, f"calling {name}", {"tool": name, "arguments": arguments})
    result = registry.call(name, context, arguments)
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
    results.append(result)
    emit(
        EventKind.TOOL_RESULT,
        result.summary if result.ok else f"{result.tool} failed: {result.error}",
        {"tool": result.tool, "ok": result.ok, "duration_ms": round(result.duration_ms, 2)},
    )
    return result


def _sample_rows(registry: ToolRegistry, context: ToolContext) -> dict[str, Any] | None:
    """Three real rows for the prompt. Best-effort: a failure here is not fatal.

    Not recorded as a step, because it is not part of the agent's reasoning — it is
    part of building the prompt, like the schema is. Showing it in "how it got there"
    would imply the agent chose to run it.
    """
    result = registry.call(
        "execute_sql", context, {"sql": f'SELECT * FROM "{TABLE_NAME}"', "max_rows": 3}
    )
    return result.data if result.ok else None


# --------------------------------------------------------------------------------
# Message plumbing
# --------------------------------------------------------------------------------


def _assistant_message(turn: ModelTurn) -> dict[str, Any]:
    """The model's own turn, replayed back to it verbatim.

    The raw tool-call payloads go back rather than the parsed ones: the conversation
    must be exactly what the model believes it said, or a model that sees its own call
    rewritten starts to distrust the transcript and repeats itself.
    """
    return {
        "role": "assistant",
        "content": turn.content,
        "tool_calls": [call.raw for call in turn.tool_calls if call.raw is not None],
    }


def _tool_message(name: str, payload: dict[str, Any]) -> dict[str, Any]:
    """One tool result, as the model sees it.

    Both `name` and `tool_name` are set. Ollama renamed this field, older builds read
    the first and newer ones the second, and a tool result the model cannot attribute
    to a call is worse than useless — it looks like an unprompted assertion.
    """
    body = json.dumps(payload, default=str)
    if len(body) > _MAX_TOOL_PAYLOAD_CHARS:
        body = (
            body[:_MAX_TOOL_PAYLOAD_CHARS]
            + " ... TRUNCATED: this result was too large to show in full. "
            + 'Rewrite the query to aggregate rather than to list rows."}'
        )
    return {"role": "tool", "name": name, "tool_name": name, "content": body}


def _call_key(name: str, arguments: dict[str, Any]) -> tuple[str, str]:
    """A stable identity for "this exact call", used to detect repeats."""
    return name, json.dumps(arguments, sort_keys=True, default=str)


def _clean(content: str) -> str:
    """Strip an inline reasoning block, if the model emitted one despite `think`."""
    return _THINK_BLOCK.sub("", content or "").strip()
