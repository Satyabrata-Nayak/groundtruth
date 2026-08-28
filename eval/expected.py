"""Compute ground truth by EXECUTING the reference SQL, and check it in.

WHY THE EXPECTED VALUES ARE NEVER TYPED BY HAND
-----------------------------------------------
A benchmark with hand-entered answers has two sources of truth that drift apart
silently. Someone tunes a generator parameter, the data moves, and forty expected
values keep asserting what used to be so. Every subsequent run measures the model
against a fossil.

Here the reference SQL is the only authored artefact, and the numbers are derived from
it. Regenerating produces a JSON diff that says exactly which answers moved -- which
is the review you want when a generator changes, and the review you never get from a
YAML file of typed constants.

WHY IT RUNS THROUGH THE SANDBOX
-------------------------------
The obvious shortcut is to query the Parquet directly with DuckDB. This deliberately
does not. Reference SQL runs through `app.data.sandbox` -- the same four layers the
agent's SQL passes -- so that a question whose answer is only reachable OUTSIDE the
sandbox fails at build time rather than at scoring time. A question the agent cannot
possibly answer is a bug in the question, and this is where it gets caught.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from app.data import sandbox
from app.tools._common import jsonable
from eval.suite import Question, Suite

# Named `answers/` rather than `expected/` on purpose: this module is
# `eval/expected.py`, and a sibling directory of the same name would sit as a
# namespace package next to a module with the identical import path. It resolves
# to the module today, and would flip the moment anyone added an __init__.py.
EXPECTED_DIR = Path(__file__).parent / "answers"
REGISTRY_PATH = EXPECTED_DIR / "registry.json"


@dataclass(frozen=True)
class ExpectedAnswer:
    """What the reference SQL produced, in the shapes the grader needs."""

    question_id: str
    columns: list[str]
    rows: list[list[Any]]
    # The named values a `facts` answer must contain: column -> value, taken from the
    # first row when there is one, or column -> [values] when the query returns several.
    facts: dict[str, Any] = field(default_factory=dict)
    # Ordered labels from the first column, for `label` and `ranking` answers.
    labels: list[str] = field(default_factory=list)
    # The single value for a `scalar` answer.
    scalar: Any = None

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, blob: dict[str, Any]) -> ExpectedAnswer:
        return cls(**blob)


def compute(question: Question, dataset_id: uuid.UUID | str, version: int) -> ExpectedAnswer:
    """Run one question's reference SQL and shape the result for grading."""
    result = sandbox.execute_sql(dataset_id, version, question.reference_sql)
    if not result.rows:
        raise ValueError(
            f"{question.id}: reference SQL returned no rows, so the question has no "
            f"ground truth. Fix the query or the question."
        )

    rows = [[jsonable(value) for value in row] for row in result.rows]
    columns = list(result.columns)

    labels = [str(row[0]) for row in rows]

    # `facts` collapses to a scalar per column when the query returns one row, and to
    # a list when it returns several -- so a two-row comparison keeps both sides.
    facts: dict[str, Any] = {}
    for position, column in enumerate(columns):
        values = [row[position] for row in rows]
        facts[column] = values[0] if len(values) == 1 else values

    scalar = rows[0][0] if len(rows) == 1 and len(columns) == 1 else None

    return ExpectedAnswer(
        question_id=question.id,
        columns=columns,
        rows=rows,
        facts=facts,
        labels=labels,
        scalar=scalar,
    )


def compute_all(
    suite: Suite, registry: dict[str, dict[str, Any]]
) -> dict[str, list[ExpectedAnswer]]:
    """Compute ground truth for every question, grouped by dataset."""
    answers: dict[str, list[ExpectedAnswer]] = {}
    for question in suite.questions:
        entry = registry.get(question.dataset)
        if entry is None:
            raise KeyError(
                f"{question.id}: dataset '{question.dataset}' is not registered. "
                f"Run `python -m eval.build` first."
            )
        answers.setdefault(question.dataset, []).append(
            compute(question, entry["dataset_id"], entry["version"])
        )
    return answers


# --------------------------------------------------------------------------------
# Persistence
# --------------------------------------------------------------------------------


def save(answers: dict[str, list[ExpectedAnswer]], directory: Path | None = None) -> list[Path]:
    """Write one JSON file per dataset. Sorted and indented, so diffs are readable."""
    directory = directory or EXPECTED_DIR
    directory.mkdir(parents=True, exist_ok=True)
    written = []
    for dataset, entries in sorted(answers.items()):
        path = directory / f"{dataset}.json"
        payload = {entry.question_id: entry.to_json() for entry in entries}
        path.write_text(
            json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
        written.append(path)
    return written


def load(directory: Path | None = None) -> dict[str, ExpectedAnswer]:
    """Read every checked-in expected-answer file, keyed by question id."""
    directory = directory or EXPECTED_DIR
    answers: dict[str, ExpectedAnswer] = {}
    for path in sorted(directory.glob("*.json")):
        if path.name == REGISTRY_PATH.name:
            continue
        for question_id, blob in json.loads(path.read_text(encoding="utf-8")).items():
            answers[question_id] = ExpectedAnswer.from_json(blob)
    return answers


def save_registry(registry: dict[str, dict[str, Any]]) -> Path:
    """Record which dataset id each evaluation dataset was registered under.

    NOT checked in: the ids are UUIDs generated on this machine, so the file is
    meaningless anywhere else. `eval.build` recreates it.
    """
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    REGISTRY_PATH.write_text(
        json.dumps(registry, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )
    return REGISTRY_PATH


def load_registry() -> dict[str, dict[str, Any]]:
    if not REGISTRY_PATH.is_file():
        raise FileNotFoundError(
            f"{REGISTRY_PATH} does not exist. Run `python -m eval.build` to generate "
            f"the evaluation datasets and register them."
        )
    return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
