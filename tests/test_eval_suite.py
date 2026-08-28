"""The question-set loader, and the real question files.

Two jobs here. First, that a malformed question file is REJECTED rather than quietly
producing a wrong score -- a benchmark that runs on a broken question set is worse
than one that refuses to run. Second, that the actual checked-in files are coherent:
ids unique, datasets known, every question backed by a stored expected answer.
"""

from __future__ import annotations

import pytest

from eval import expected as expected_module
from eval.datasets import BY_NAME
from eval.suite import QUESTIONS_DIR, QuestionSetError, load_suite


def write_suite(tmp_path, body: str):
    path = tmp_path / "sample.yaml"
    path.write_text(body, encoding="utf-8")
    return tmp_path


VALID = """
dataset: ecommerce
questions:
  - id: x-1
    category: lookup
    difficulty: easy
    question: How many rows?
    reference_sql: |
      SELECT count(*) FROM dataset
    answer: {kind: scalar, tolerance: 0}
"""


# ------------------------------------------------------------------------ validation


def test_valid_file_loads(tmp_path):
    suite = load_suite(write_suite(tmp_path, VALID))
    assert len(suite.questions) == 1
    assert suite.questions[0].id == "x-1"
    assert suite.questions[0].min_tool_calls == 1


def test_missing_dataset_is_rejected(tmp_path):
    with pytest.raises(QuestionSetError, match="missing top-level 'dataset'"):
        load_suite(write_suite(tmp_path, "questions: []"))


def test_unknown_category_is_rejected(tmp_path):
    body = VALID.replace("category: lookup", "category: vibes")
    with pytest.raises(QuestionSetError, match="category 'vibes'"):
        load_suite(write_suite(tmp_path, body))


def test_unknown_answer_kind_is_rejected(tmp_path):
    body = VALID.replace("kind: scalar", "kind: telepathy")
    with pytest.raises(QuestionSetError, match="answer.kind"):
        load_suite(write_suite(tmp_path, body))


def test_ranking_without_top_k_is_rejected(tmp_path):
    """A ranking with no declared depth cannot be graded, so it must not load."""
    body = VALID.replace("kind: scalar", "kind: ranking")
    with pytest.raises(QuestionSetError, match="needs answer.top_k"):
        load_suite(write_suite(tmp_path, body))


def test_missing_reference_sql_is_rejected(tmp_path):
    body = VALID.replace("      SELECT count(*) FROM dataset\n", "")
    with pytest.raises(QuestionSetError, match="missing 'reference_sql'"):
        load_suite(write_suite(tmp_path, body))


def test_duplicate_ids_are_rejected(tmp_path):
    duplicated = VALID + VALID.split("questions:")[1]
    with pytest.raises(QuestionSetError, match="duplicate question id"):
        load_suite(write_suite(tmp_path, duplicated))


def test_empty_directory_is_rejected(tmp_path):
    with pytest.raises(QuestionSetError, match="no question files"):
        load_suite(tmp_path)


# --------------------------------------------------------------- the real question set


@pytest.fixture(scope="module")
def suite():
    return load_suite()


def test_the_real_suite_loads(suite):
    assert len(suite.questions) >= 40


def test_every_question_targets_a_known_dataset(suite):
    for question in suite.questions:
        assert question.dataset in BY_NAME, f"{question.id} names an unknown dataset"


def test_every_dataset_has_questions(suite):
    covered = {q.dataset for q in suite.questions}
    assert covered == set(BY_NAME)


def test_reference_sql_is_read_only(suite):
    """Every reference query must be executable by the agent's own sandbox."""
    from app.data.sandbox import validate_sql

    for question in suite.questions:
        validate_sql(question.reference_sql)


def test_every_category_is_represented(suite):
    counts = suite.categories()
    for category in ("lookup", "aggregation", "comparison", "trend", "diagnosis",
                     "data_quality"):
        assert counts.get(category, 0) > 0, f"no questions in category {category}"


def test_hard_questions_demand_more_than_one_call(suite):
    """A 'hard' question answerable in one call is mislabelled, not hard."""
    diagnosis = [q for q in suite.questions if q.category == "diagnosis"]
    assert diagnosis
    assert all(q.min_tool_calls >= 2 for q in diagnosis)


def test_expected_answers_exist_for_every_question(suite):
    """The checked-in ground truth must cover the checked-in questions.

    This is the test that catches "someone added a question and forgot to run
    `python -m eval.build`", which would otherwise surface as a crash mid-benchmark.
    """
    stored = expected_module.load()
    if not stored:
        pytest.skip("no expected answers checked in yet")
    missing = [q.id for q in suite.questions if q.id not in stored]
    assert not missing, f"run `python -m eval.build`; missing: {missing}"


def test_question_files_cover_the_files_on_disk():
    names = {path.stem for path in QUESTIONS_DIR.glob("*.yaml")}
    assert names == set(BY_NAME)
