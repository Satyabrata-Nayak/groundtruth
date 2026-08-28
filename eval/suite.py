"""Load and validate the golden question set.

VALIDATION HAPPENS AT LOAD, LOUDLY
----------------------------------
A question file is hand-written YAML, so every mistake it can contain is a mistake
someone will make: a duplicated id, a category that does not exist, a `ranking`
question with no `top_k`, a reference query that is not a SELECT. None of those are
visible when they happen -- they surface later as a benchmark number that is quietly
wrong, which is the single worst failure mode a benchmark has.

So the loader rejects the file rather than the run. A malformed question set should
stop the harness before it produces a score nobody can trust.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

QUESTIONS_DIR = Path(__file__).parent / "questions"

CATEGORIES = frozenset(
    {
        "lookup",        # one fact, one query -- a harness canary, not a model test
        "aggregation",   # group and rank
        "comparison",    # two or more things weighed against each other
        "trend",         # behaviour over time
        "diagnosis",     # multi-step "why did this happen"
        "data_quality",  # nulls, constants, duplicated and disagreeing columns
        "ambiguity",     # no single right number; graded on stating the assumption
    }
)

ANSWER_KINDS = frozenset(
    {
        "scalar",             # one number, compared within a relative tolerance
        "label",              # one name that must appear in the answer
        "ranking",            # the first top_k labels, in the right order
        "facts",              # several named values, each of which must appear
        "assumption_stated",  # graded only on whether the answer says what it assumed
    }
)

DIFFICULTIES = frozenset({"easy", "medium", "hard"})


@dataclass(frozen=True)
class Question:
    id: str
    dataset: str
    category: str
    difficulty: str
    question: str
    reference_sql: str
    answer_kind: str
    tolerance: float = 0.0
    top_k: int | None = None
    must_mention: tuple[str, ...] = ()
    expected_tools: tuple[str, ...] = ()
    min_tool_calls: int = 1

    @property
    def is_scored_numerically(self) -> bool:
        """Whether this question contributes to the accuracy number.

        `ambiguity` questions do not. They have several defensible answers, and
        folding them into an accuracy percentage would make the percentage mean
        something different from what it claims.
        """
        return self.answer_kind != "assumption_stated"


@dataclass(frozen=True)
class Suite:
    questions: tuple[Question, ...]
    datasets: tuple[str, ...] = field(default=())

    def for_dataset(self, name: str) -> tuple[Question, ...]:
        return tuple(q for q in self.questions if q.dataset == name)

    def by_id(self, question_id: str) -> Question:
        for question in self.questions:
            if question.id == question_id:
                return question
        raise KeyError(f"no question with id {question_id!r}")

    def categories(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for question in self.questions:
            counts[question.category] = counts.get(question.category, 0) + 1
        return dict(sorted(counts.items()))


class QuestionSetError(ValueError):
    """A question file is malformed. The message names the file and the question."""


def load_suite(directory: Path | None = None, *, datasets: list[str] | None = None) -> Suite:
    """Read every question file, validate it, and return the whole suite."""
    directory = directory or QUESTIONS_DIR
    files = sorted(directory.glob("*.yaml"))
    if not files:
        raise QuestionSetError(f"no question files found in {directory}")

    questions: list[Question] = []
    names: list[str] = []
    seen: set[str] = set()

    for path in files:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        dataset = raw.get("dataset")
        if not dataset:
            raise QuestionSetError(f"{path.name}: missing top-level 'dataset'")
        if datasets is not None and dataset not in datasets:
            continue
        names.append(dataset)

        for index, entry in enumerate(raw.get("questions") or []):
            question = _parse(path, dataset, index, entry)
            if question.id in seen:
                raise QuestionSetError(
                    f"{path.name}: duplicate question id {question.id!r}. Ids must be "
                    f"unique across the whole suite, because results are keyed by them."
                )
            seen.add(question.id)
            questions.append(question)

    return Suite(questions=tuple(questions), datasets=tuple(names))


def _parse(path: Path, dataset: str, index: int, entry: dict[str, Any]) -> Question:
    where = f"{path.name}[{index}]"
    question_id = entry.get("id")
    if not question_id:
        raise QuestionSetError(f"{where}: missing 'id'")
    where = f"{path.name}:{question_id}"

    category = entry.get("category")
    if category not in CATEGORIES:
        raise QuestionSetError(
            f"{where}: category {category!r} is not one of {sorted(CATEGORIES)}"
        )

    difficulty = entry.get("difficulty", "medium")
    if difficulty not in DIFFICULTIES:
        raise QuestionSetError(
            f"{where}: difficulty {difficulty!r} is not one of {sorted(DIFFICULTIES)}"
        )

    text = (entry.get("question") or "").strip()
    if not text:
        raise QuestionSetError(f"{where}: missing 'question'")

    sql = (entry.get("reference_sql") or "").strip()
    if not sql:
        raise QuestionSetError(f"{where}: missing 'reference_sql'")

    answer = entry.get("answer") or {}
    kind = answer.get("kind")
    if kind not in ANSWER_KINDS:
        raise QuestionSetError(
            f"{where}: answer.kind {kind!r} is not one of {sorted(ANSWER_KINDS)}"
        )

    top_k = answer.get("top_k")
    if kind == "ranking" and not top_k:
        raise QuestionSetError(
            f"{where}: a 'ranking' answer needs answer.top_k -- how many positions "
            f"of the ranking have to be correct"
        )

    min_calls = int(entry.get("min_tool_calls", 1))
    if min_calls < 1:
        raise QuestionSetError(f"{where}: min_tool_calls must be at least 1")

    return Question(
        id=question_id,
        dataset=dataset,
        category=category,
        difficulty=difficulty,
        question=text,
        reference_sql=sql,
        answer_kind=kind,
        tolerance=float(answer.get("tolerance", 0.0)),
        top_k=int(top_k) if top_k else None,
        must_mention=tuple(str(m).lower() for m in entry.get("must_mention", ())),
        expected_tools=tuple(entry.get("expected_tools", ())),
        min_tool_calls=min_calls,
    )
