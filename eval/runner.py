"""Run an agent against the golden question set and report a score.

    python -m eval.runner --agent oracle          the top of the scale (expect ~100%)
    python -m eval.runner --agent refusing        the bottom of the scale (expect 0%)
    python -m eval.runner --agent schema-only     plausible but uncomputed (expect ~0%)
    python -m eval.runner --dataset ecommerce     one dataset
    python -m eval.runner --category diagnosis    one question type
    python -m eval.runner --json results.json     machine-readable output

WHAT THE REPORT SEPARATES, AND WHY
----------------------------------
    values      the numbers and names the answer had to contain
    mentions    the concepts a correct explanation had to name
    tools       calls attempted, and how many failed

They are reported apart because they fail for different reasons and are fixed in
different places. An agent with correct values and missing mentions needs a prompt
change; one with failing tool calls needs better tool descriptions or error messages;
one with wrong values needs neither, it needs a better query.

`ambiguity` questions are counted separately and never folded into the headline
accuracy. They have several defensible answers, so including them would make the
percentage mean something other than what it says.

CALIBRATION, AND ONE PLACE IT DOES NOT REACH
--------------------------------------------
Three stub agents bracket the scale, so a real score has known endpoints:

    oracle       100% accuracy, 100% values   -- the grader's ceiling is real
    refusing       0% accuracy,   0% values   -- no free passes for saying nothing
    schema-only    0% accuracy,   0% values   -- no free passes for sounding right

That third one matters most: it inspects the schema and writes a fluent, confident
answer containing no computed number. It scores zero on values, which is the property
that makes this benchmark worth running.

The exception is the `ambiguity` category, where the oracle also scores 0/2. Its
check -- did the answer say which column it used -- can only be satisfied by narrative
behaviour, and a stub that renders a result table never produces any. So that
category's ceiling is UNVERIFIED, and it is excluded from headline accuracy for that
reason as much as for having several defensible answers. The first real agent in M5 is
what will establish whether the check is reasonable.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from app.tools import ToolContext, get_registry
from eval import expected as expected_module
from eval.agents import BUILTIN_AGENTS, Agent, AgentRun
from eval.grader import Grade, grade, summarise
from eval.suite import Question, Suite, load_suite


def log(message: str = "") -> None:
    print(message, flush=True)


def run_agent(
    agent: Agent,
    suite: Suite,
    registry_entries: dict[str, dict[str, Any]],
    expected_answers: dict[str, expected_module.ExpectedAnswer],
    *,
    verbose: bool = False,
) -> list[Grade]:
    tools = get_registry()
    grades: list[Grade] = []

    for index, question in enumerate(suite.questions, start=1):
        entry = registry_entries[question.dataset]
        context = ToolContext(dataset_id=entry["dataset_id"], version=entry["version"])

        started = time.perf_counter()
        try:
            run = agent.answer(question, context, tools)
        except Exception as exc:  # noqa: BLE001 - an agent crash is a failing question
            run = AgentRun(answer="", duration_s=time.perf_counter() - started, error=str(exc))

        result = grade(
            question,
            expected_answers[question.id],
            run.answer,
            tool_calls=run.tool_calls,
            failed_calls=run.failed_calls,
            duration_s=run.duration_s,
            unknown_tools=run.unknown_tools,
        )
        grades.append(result)

        mark = "PASS" if result.correct else "FAIL"
        log(
            f"[{index:>2}/{len(suite.questions)}] {mark}  {question.id:<11} "
            f"{question.category:<13} {run.tool_calls} call(s)  {run.duration_s * 1000:>7.0f} ms"
        )
        if verbose and not result.correct:
            for reason in result.reasons:
                log(f"            - {reason}")
            log(f"            answer: {run.answer[:200]!r}")

    return grades


def report(agent_name: str, suite: Suite, grades: list[Grade]) -> dict[str, Any]:
    """Print the scoreboard and return it as data."""
    scored = [g for g in grades if suite.by_id(g.question_id).is_scored_numerically]
    ambiguous = [g for g in grades if not suite.by_id(g.question_id).is_scored_numerically]

    stats = summarise(scored)
    total = stats["total"]

    log("\n" + "=" * 68)
    log(f"agent: {agent_name}")
    log("=" * 68)
    log(f"  accuracy          {stats['correct']}/{total}  ({stats['correct'] / total:.1%})")
    log(
        f"  values correct    {stats['values_correct']}/{total}  "
        f"({stats['values_correct'] / total:.1%})"
    )
    mention_total = stats["mentions_total"]
    mention_share = (
        f"({stats['mentions_present'] / mention_total:.1%})" if mention_total else "(n/a)"
    )
    log(f"  mentions present  {stats['mentions_present']}/{mention_total}  {mention_share}")
    log(f"  tool calls        {stats['tool_calls']} ({stats['failed_calls']} failed)")
    log(f"  wall clock        {stats['duration_s']:.2f}s")

    log("\n  by category")
    for category, bucket in stats["by_category"].items():
        share = bucket["correct"] / bucket["total"]
        bar = "#" * round(share * 20)
        log(f"    {category:<14} {bucket['correct']:>2}/{bucket['total']:<2} {bar:<20} {share:.0%}")

    if ambiguous:
        passed = sum(g.correct for g in ambiguous)
        log(f"\n  ambiguity (reported separately)  {passed}/{len(ambiguous)}")

    failures = [g for g in scored if not g.correct]
    if failures:
        log(f"\n  failed: {', '.join(g.question_id for g in failures)}")

    return {
        "agent": agent_name,
        "summary": stats,
        "ambiguity": {
            "total": len(ambiguous),
            "passed": sum(g.correct for g in ambiguous),
        },
        "questions": [
            {
                "id": g.question_id,
                "category": g.category,
                "correct": g.correct,
                "values_correct": g.values_correct,
                "mentions_present": g.mentions_present,
                "tool_calls": g.tool_calls,
                "failed_calls": g.failed_calls,
                "duration_s": round(g.duration_s, 4),
                "reasons": g.reasons,
                "missing_values": g.missing_values,
                "missing_mentions": g.missing_mentions,
            }
            for g in grades
        ],
    }


def _filter(suite: Suite, dataset: str | None, category: str | None) -> Suite:
    questions: tuple[Question, ...] = suite.questions
    if dataset:
        questions = tuple(q for q in questions if q.dataset == dataset)
    if category:
        questions = tuple(q for q in questions if q.category == category)
    if not questions:
        raise SystemExit("no questions matched those filters")
    return Suite(questions=questions, datasets=suite.datasets)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Score an agent against the golden question set.")
    parser.add_argument("--agent", default="oracle", help=f"one of {sorted(BUILTIN_AGENTS)}")
    parser.add_argument("--dataset", help="restrict to one evaluation dataset")
    parser.add_argument("--category", help="restrict to one question category")
    parser.add_argument("--json", type=Path, help="write the full result to this file")
    parser.add_argument("-v", "--verbose", action="store_true", help="explain each failure")
    args = parser.parse_args(argv)

    if args.agent not in BUILTIN_AGENTS:
        log(f"unknown agent '{args.agent}'. Available: {', '.join(sorted(BUILTIN_AGENTS))}")
        return 2

    try:
        entries = expected_module.load_registry()
    except FileNotFoundError as exc:
        log(str(exc))
        return 2

    expected_answers = expected_module.load()
    if not expected_answers:
        log("no expected answers found. Run `python -m eval.build` first.")
        return 2

    suite = _filter(load_suite(), args.dataset, args.category)
    missing = [q.id for q in suite.questions if q.id not in expected_answers]
    if missing:
        log(
            f"{len(missing)} question(s) have no expected answer: {', '.join(missing[:5])}. "
            f"Run `python -m eval.build`."
        )
        return 2

    agent = BUILTIN_AGENTS[args.agent]()
    log(f"running {len(suite.questions)} question(s) against agent '{agent.name}'\n")

    grades = run_agent(agent, suite, entries, expected_answers, verbose=args.verbose)
    payload = report(agent.name, suite, grades)

    if args.json:
        args.json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        log(f"\nwrote {args.json}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
