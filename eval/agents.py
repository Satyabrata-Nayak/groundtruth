"""The agent interface the runner scores, and the two stubs that calibrate the runner.

WHY STUB AGENTS EXIST BEFORE A REAL ONE
---------------------------------------
M1 produced five "model failures" that were all bugs in the measuring harness. The
lesson was not "be careful"; it was that an instrument has to be checked against known
inputs before its readings mean anything. A benchmark reporting 62% is useless unless
you know the harness would have said 100% for a perfect answer and 0% for a worthless
one.

So the scoreboard ships with both ends of its own scale:

    OracleAgent    executes the question's own reference SQL and reports the result.
                   It has the right answer by construction. If it does not score at
                   or near 100%, THE GRADER IS BROKEN -- not the agent.

    RefusingAgent  answers "I don't know" and calls nothing. If it scores above zero,
                   the grader is handing out free passes, and every later number is
                   inflated by however many it gave away.

`LocalModelAgent` is the real one, and it slots into the same interface. When it
scores 62%, that 62% sits on a scale whose endpoints have been measured.

THE ORACLE IS NOT A CHEAT
-------------------------
It reads `reference_sql` from the question, which no real agent may do. That is the
point: it isolates the grader from the agent. Its failures are grader bugs, and the
runner labels its column accordingly.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

from app.tools import ToolContext, ToolRegistry, ToolResult
from eval.suite import Question

if TYPE_CHECKING:  # pragma: no cover - import cost only matters at run time
    from app.agent.llm import LlmClient


@dataclass
class AgentRun:
    """What an agent did and what it concluded."""

    answer: str
    tool_results: list[ToolResult] = field(default_factory=list)
    duration_s: float = 0.0
    error: str | None = None

    @property
    def tool_calls(self) -> int:
        return len(self.tool_results)

    @property
    def failed_calls(self) -> int:
        return sum(1 for r in self.tool_results if not r.ok)

    @property
    def unknown_tools(self) -> list[str]:
        """Tools the agent tried to call that do not exist.

        Tracked separately from other failures because it is a different kind of
        mistake: a bad argument is a repairable slip, an invented tool name means the
        agent is not working from the action space it was given.
        """
        return sorted(
            {
                r.tool
                for r in self.tool_results
                if not r.ok and r.error and r.error.startswith("unknown tool")
            }
        )


class Agent(Protocol):
    """Anything the runner can score."""

    name: str

    def answer(
        self, question: Question, context: ToolContext, registry: ToolRegistry
    ) -> AgentRun: ...


class OracleAgent:
    """Answers from the question's own reference SQL. Calibrates the top of the scale."""

    name = "oracle"

    def answer(self, question: Question, context: ToolContext, registry: ToolRegistry) -> AgentRun:
        started = time.perf_counter()
        result = registry.call(
            "execute_sql", context, {"sql": question.reference_sql, "max_rows": 50}
        )
        duration = time.perf_counter() - started

        if not result.ok:
            return AgentRun(
                answer="",
                tool_results=[result],
                duration_s=duration,
                error=result.error,
            )

        return AgentRun(
            answer=_render(result.data),
            tool_results=[result],
            duration_s=duration,
        )


class RefusingAgent:
    """Answers nothing useful and calls nothing. Calibrates the bottom of the scale."""

    name = "refusing"

    def answer(self, question: Question, context: ToolContext, registry: ToolRegistry) -> AgentRun:
        return AgentRun(answer="I don't have enough information to answer that.")


class SchemaOnlyAgent:
    """Inspects the schema, then answers without querying anything.

    A third calibration point, and the most realistic failure mode of a weak agent:
    it does real work, produces a fluent answer, and states nothing that was computed.
    Its score is roughly the share of questions a benchmark would pass on plausibility
    alone -- which should be close to zero, and is worth knowing rather than assuming.
    """

    name = "schema-only"

    def answer(self, question: Question, context: ToolContext, registry: ToolRegistry) -> AgentRun:
        started = time.perf_counter()
        result = registry.call("inspect_schema", context, {})
        columns = ", ".join(c["name"] for c in result.data.get("columns", [])) if result.ok else ""
        return AgentRun(
            answer=(
                f"Looking at the dataset, the relevant columns are {columns}. "
                f"The data suggests a clear pattern in response to: {question.question}"
            ),
            tool_results=[result],
            duration_s=time.perf_counter() - started,
        )


class _RecordingRegistry:
    """A registry that remembers every result, so the runner can score HOW as well as WHAT.

    `Registry.call` is the single choke point between a model and any computation, so
    wrapping it is enough — there is no second path to record. The agent loop is
    handed this instead of the real registry and cannot tell the difference.
    """

    def __init__(self, inner: ToolRegistry) -> None:
        self._inner = inner
        self.results: list[ToolResult] = []

    def call(self, name: str, context: ToolContext, arguments: dict[str, Any] | None = None):
        result = self._inner.call(name, context, arguments)
        self.results.append(result)
        return result

    def get(self, name: str):
        return self._inner.get(name)

    def names(self) -> list[str]:
        return self._inner.names()

    def specs(self, only: list[str] | None = None) -> list[dict[str, Any]]:
        return self._inner.specs(only)


class LocalModelAgent:
    """The real agent: a local model choosing tools, scored on the same scale as the stubs.

    It runs the SAME `run_agent_analysis` the worker runs — not a copy of it tuned for
    the benchmark. A harness that measures a special evaluation path measures the
    harness.
    """

    name = "local-model"

    def __init__(self, client: LlmClient | None = None) -> None:
        self._client = client

    def answer(self, question: Question, context: ToolContext, registry: ToolRegistry) -> AgentRun:
        from app.agent.analyst import run_agent_analysis
        from app.agent.contract import AnalysisFailed

        recording = _RecordingRegistry(registry)
        started = time.perf_counter()
        try:
            result = run_agent_analysis(
                dataset_id=context.dataset_id,
                version=context.version,
                question=question.question,
                emit=lambda *args, **kwargs: None,
                checkpoint=lambda: None,
                client=self._client,
                registry=recording,
            )
        except AnalysisFailed as exc:
            return AgentRun(
                answer="",
                tool_results=recording.results,
                duration_s=time.perf_counter() - started,
                error=str(exc),
            )

        return AgentRun(
            answer=result["answer"],
            tool_results=recording.results,
            duration_s=time.perf_counter() - started,
        )


def _render(data: dict[str, Any]) -> str:
    """Turn a query result into readable text.

    Column names are included, not just values: the grader's `must_mention` checks look
    for concepts ("discount", "electronics"), and in a reference result those concepts
    are usually carried by the column names rather than by the numbers.
    """
    columns = data.get("columns", [])
    rows = data.get("rows", [])
    if not rows:
        return "The query returned no rows."

    lines = [
        ", ".join(f"{column} = {value}" for column, value in zip(columns, row, strict=False))
        for row in rows
    ]
    return "Result:\n" + "\n".join(lines)


BUILTIN_AGENTS: dict[str, type] = {
    "oracle": OracleAgent,
    "refusing": RefusingAgent,
    "schema-only": SchemaOnlyAgent,
    "local-model": LocalModelAgent,
}
