"""The grader. Every benchmark number this project ever reports passes through here.

The tests are weighted toward FALSE PASSES rather than false failures. A grader that
is too strict produces a visibly bad score somebody investigates; a grader that is too
lenient produces a good score nobody questions, and that is how a benchmark quietly
stops measuring anything.
"""

from __future__ import annotations

import pytest

from eval.expected import ExpectedAnswer
from eval.grader import ExtractedNumber, extract_numbers, grade, number_matches
from eval.suite import Question


def make_question(**overrides) -> Question:
    defaults = dict(
        id="q-1",
        dataset="ecommerce",
        category="aggregation",
        difficulty="medium",
        question="?",
        reference_sql="SELECT 1",
        answer_kind="scalar",
        tolerance=0.01,
    )
    defaults.update(overrides)
    return Question(**defaults)


def make_expected(**overrides) -> ExpectedAnswer:
    defaults = dict(
        question_id="q-1",
        columns=["value"],
        rows=[[42.0]],
        facts={"value": 42.0},
        labels=["42.0"],
        scalar=42.0,
    )
    defaults.update(overrides)
    return ExpectedAnswer(**defaults)


# ------------------------------------------------------------------ number extraction


@pytest.mark.parametrize(
    "text,expected",
    [
        ("the total is 1234", 1234.0),
        ("the total is 1,234.56", 1234.56),
        ("revenue was $2,501,150.87", 2501150.87),
        ("about 2.5 million", 2_500_000.0),
        ("roughly 15k orders", 15_000.0),
        ("a margin of -0.075", -0.075),
        ("it rose to .5", 0.5),
    ],
)
def test_numbers_are_extracted_from_prose(text, expected):
    values = [n.value for n in extract_numbers(text)]
    assert expected in values


def test_percent_tokens_are_flagged():
    found = extract_numbers("the return rate is 15.31%")
    assert found[0].was_percent is True


def test_word_percent_is_flagged_too():
    found = extract_numbers("the return rate is 15.31 percent")
    assert found[0].was_percent is True


# --------------------------------------------------------------------- number matching


def test_exact_match():
    assert number_matches(42.0, extract_numbers("the answer is 42"), 0.0)


def test_relative_tolerance_is_applied():
    numbers = extract_numbers("about 101")
    assert number_matches(100.0, numbers, 0.02)
    assert not number_matches(100.0, numbers, 0.005)


def test_percent_form_matches_a_fraction():
    """0.1531 and '15.31%' are the same quantity."""
    assert number_matches(0.1531, extract_numbers("the rate is 15.31%"), 0.01)


def test_fraction_form_matches_a_fraction():
    assert number_matches(0.1531, extract_numbers("the rate is 0.1531"), 0.01)


def test_bare_number_rescales_only_when_truth_is_a_rate():
    """'15.31' with no % sign may mean 0.1531 -- but only where truth is a rate."""
    assert number_matches(0.1531, extract_numbers("the rate is 15.31"), 0.01)


def test_rescaling_cannot_manufacture_a_false_pass():
    """An answer of '1' must NOT satisfy an expected value of 100.

    This is the guard that stops the percent handling from turning into a free pass
    on every large number.
    """
    assert not number_matches(100.0, extract_numbers("the answer is 1"), 0.01)


def test_zero_expected_uses_absolute_tolerance():
    assert number_matches(0.0, [ExtractedNumber(0.0, False)], 0.0)
    assert not number_matches(0.0, [ExtractedNumber(5.0, False)], 0.01)


# ---------------------------------------------------------------------------- scalar


def test_scalar_pass():
    result = grade(make_question(), make_expected(), "The total is 42.")
    assert result.correct


def test_scalar_fail_reports_the_missing_value():
    result = grade(make_question(), make_expected(), "The total is 99.")
    assert not result.correct
    assert "42.0" in result.missing_values


def test_empty_answer_fails():
    result = grade(make_question(), make_expected(), "")
    assert not result.correct
    assert "no answer text produced" in result.reasons


# ----------------------------------------------------------------------------- label


def test_label_match_is_case_insensitive():
    question = make_question(answer_kind="label")
    expected = make_expected(labels=["Books"], rows=[["Books"]], facts={"c": "Books"}, scalar=None)
    assert grade(question, expected, "The best category is books.").correct


def test_label_respects_word_boundaries():
    """'S-05' must not be satisfied by 'S-050'."""
    question = make_question(answer_kind="label")
    expected = make_expected(labels=["S-05"], rows=[["S-05"]], facts={"s": "S-05"}, scalar=None)
    assert not grade(question, expected, "sensor S-050 is the culprit").correct
    assert grade(question, expected, "sensor S-05 is the culprit").correct


def test_label_with_punctuation_is_matched_literally():
    question = make_question(answer_kind="label")
    expected = make_expected(
        labels=["Home & Garden"],
        rows=[["Home & Garden"]],
        facts={"c": "Home & Garden"},
        scalar=None,
    )
    assert grade(question, expected, "Home & Garden leads.").correct


# --------------------------------------------------------------------------- ranking


def _ranking_case():
    question = make_question(answer_kind="ranking", top_k=3, tolerance=0.0)
    expected = make_expected(
        labels=["Partner", "Online", "Retail"],
        rows=[["Partner"], ["Online"], ["Retail"]],
        facts={"channel": ["Partner", "Online", "Retail"]},
        scalar=None,
    )
    return question, expected


def test_ranking_in_order_passes():
    question, expected = _ranking_case()
    assert grade(question, expected, "Partner, then Online, then Retail.").correct


def test_ranking_out_of_order_fails():
    """Naming the right three in the wrong order is not a ranking."""
    question, expected = _ranking_case()
    result = grade(question, expected, "Retail, then Online, then Partner.")
    assert not result.correct
    assert any("order is wrong" in reason for reason in result.reasons)


def test_ranking_missing_an_entry_fails():
    question, expected = _ranking_case()
    result = grade(question, expected, "Partner and then Online.")
    assert not result.correct
    assert "Retail" in result.missing_values


# ----------------------------------------------------------------------------- facts


def test_facts_requires_every_value():
    question = make_question(answer_kind="facts", tolerance=0.01)
    expected = make_expected(
        columns=["region", "revenue", "margin"],
        rows=[["West", 891913.42, 0.0491]],
        facts={"region": "West", "revenue": 891913.42, "margin": 0.0491},
        labels=["West"],
        scalar=None,
    )
    full = "West leads with revenue of 891,913.42 but a margin of only 4.91%."
    assert grade(question, expected, full).correct

    partial = "West leads with revenue of 891,913.42."
    result = grade(question, expected, partial)
    assert not result.correct
    assert any("margin" in miss for miss in result.missing_values)


def test_facts_handles_multi_row_lists():
    question = make_question(answer_kind="facts", tolerance=0.01)
    expected = make_expected(
        columns=["quarter", "revenue"],
        rows=[[2, 576382.88], [3, 793071.25]],
        facts={"quarter": [2, 3], "revenue": [576382.88, 793071.25]},
        labels=["2", "3"],
        scalar=None,
    )
    text = "Q2 revenue was 576,382.88 and Q3 revenue was 793,071.25."
    assert grade(question, expected, text).correct


# ---------------------------------------------------------------------- must_mention


def test_must_mention_blocks_a_numerically_correct_answer():
    """Right numbers with no explanation is a real failure mode, tracked separately."""
    question = make_question(must_mention=("discount",))
    result = grade(question, make_expected(), "The total is 42.")
    assert result.values_correct
    assert not result.mentions_present
    assert not result.correct
    assert "discount" in result.missing_mentions


def test_must_mention_accepts_synonyms():
    question = make_question(must_mention=("differ|disagree|not the same",))
    assert grade(question, make_expected(), "42; the two columns disagree.").correct


def test_mention_requirement_is_flagged_for_the_denominator():
    with_requirement = grade(make_question(must_mention=("x",)), make_expected(), "42")
    without = grade(make_question(), make_expected(), "42")
    assert with_requirement.has_mention_requirement
    assert not without.has_mention_requirement


# ------------------------------------------------------------------- assumption_stated


def test_assumption_question_needs_an_explicit_marker():
    question = make_question(answer_kind="assumption_stated", must_mention=("conversions",))
    bare = grade(question, make_expected(), "The conversions rate is 3.5%.")
    assert not bare.correct

    stated = grade(question, make_expected(), "Using the conversions column, the rate is 3.5%.")
    assert stated.correct


# --------------------------------------------------------------------------- tooling


def test_invented_tool_names_fail_the_question():
    result = grade(
        make_question(),
        make_expected(),
        "The total is 42.",
        unknown_tools=["run_python"],
    )
    assert result.values_correct
    assert not result.tools_valid
    assert not result.correct
