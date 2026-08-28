"""Score one answer against ground truth.

THE HARD PART IS NOT COMPARING NUMBERS, IT IS FINDING THEM
-----------------------------------------------------------
Ground truth is `0.1531`. A correct answer might say any of:

    "15.31%"      "about 15%"      "0.1531"      "15.3 percent"
    "1,531 per 10,000"             "roughly one in seven"

A grader that only accepts the literal digits marks most correct answers wrong, and a
benchmark that under-reports is worse than no benchmark: it sends you optimising a
model that was already right. So numbers are extracted from free text, normalised
(commas, currency symbols, k/M/B suffixes, percent signs) and compared within a
relative tolerance the question declares.

PERCENT HANDLING IS THE SUBTLE PART
-----------------------------------
`15.31%` and `0.1531` are the same quantity. Rescaling by 100 is therefore allowed --
but only where it cannot manufacture a false pass:

    the token carried a % sign        -> try it as both a percentage and a fraction
    ground truth is a rate (|x| < 1)  -> try the answer divided by 100
    otherwise                         -> compare at face value only

Without that last restriction, an expected value of 100 would be satisfied by an
answer containing "1", which is exactly the kind of silent free pass that makes a
scoreboard lie.

WHY VALUES AND MENTIONS ARE SCORED SEPARATELY
---------------------------------------------
A diagnosis question has two failure modes: the numbers can be wrong, or the numbers
can be right with no explanation of what caused them. Collapsing both into one boolean
loses the distinction precisely where it matters most, so `Grade` carries both and the
report shows them side by side.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from eval.expected import ExpectedAnswer
from eval.suite import Question

# A number, with optional sign, thousands separators, decimals and a magnitude or
# percent suffix. Currency symbols are stripped before matching.
#
# THE LOOKBEHIND EXCLUDES DIGITS AND DOTS, NOT ALL WORD CHARACTERS.
# The stricter `(?<![\w.])` rejects any digit preceded by a letter, which silently
# made "Q3", "H1" and "S-07" unmatchable -- and quarters are exactly how a real answer
# refers to a quarter. Excluding only digits and dots keeps what the guard was
# actually for (not re-matching the "456" inside "123.456", not splitting "1,234"),
# while letting a letter-prefixed number through.
_NUMBER = re.compile(
    r"(?<![\d.])"
    r"(-?\d{1,3}(?:,\d{3})+(?:\.\d+)?|-?\d+(?:\.\d+)?|-?\.\d+)"
    r"\s*(%|percent|k\b|m\b|bn?\b|million|billion|thousand)?",
    re.IGNORECASE,
)

_MAGNITUDES = {
    "k": 1e3,
    "thousand": 1e3,
    "m": 1e6,
    "million": 1e6,
    "b": 1e9,
    "bn": 1e9,
    "billion": 1e9,
}

# Default relative tolerance when a question does not set one. Tight enough that a
# wrong query fails, loose enough that sensible rounding does not.
DEFAULT_TOLERANCE = 0.01

# Phrases that show an answer has declared its interpretation rather than silently
# picking one. Deliberately broad -- the requirement is that a choice was made
# visible, not that it was announced with any particular formula.
_ASSUMPTION_MARKERS = (
    "assum",
    "i used",
    "using the",
    "interpret",
    "based on the",
    "taking ",
    "defined as",
    "calculated from",
    "i have taken",
)


@dataclass(frozen=True)
class ExtractedNumber:
    value: float
    was_percent: bool


@dataclass
class Grade:
    question_id: str
    category: str
    # Did every value the answer had to contain actually appear?
    values_correct: bool = False
    # Did the answer name the concepts a correct explanation requires?
    mentions_present: bool = True
    # Did the run use only tools that exist, and succeed at them?
    tools_valid: bool = True
    tool_calls: int = 0
    failed_calls: int = 0
    duration_s: float = 0.0
    reasons: list[str] = field(default_factory=list)
    missing_values: list[str] = field(default_factory=list)
    missing_mentions: list[str] = field(default_factory=list)
    # Whether this question asked for any concept to be named at all.
    has_mention_requirement: bool = False

    @property
    def correct(self) -> bool:
        return self.values_correct and self.mentions_present and self.tools_valid


def extract_numbers(text: str) -> list[ExtractedNumber]:
    """Every number in a block of free text, normalised to a float."""
    cleaned = text.replace("$", " ").replace("£", " ").replace("€", " ")
    found: list[ExtractedNumber] = []
    for match in _NUMBER.finditer(cleaned):
        raw, suffix = match.group(1), (match.group(2) or "").lower().strip()
        try:
            value = float(raw.replace(",", ""))
        except ValueError:
            continue
        is_percent = suffix in ("%", "percent")
        if suffix in _MAGNITUDES:
            value *= _MAGNITUDES[suffix]
        found.append(ExtractedNumber(value=value, was_percent=is_percent))
    return found


def number_matches(expected: float, candidates: list[ExtractedNumber], tolerance: float) -> bool:
    """Whether any number in the answer equals `expected` within a relative tolerance."""
    tolerance = tolerance if tolerance > 0 else 0.0
    expected_is_rate = abs(expected) < 1

    for candidate in candidates:
        forms = [candidate.value]
        # A percent-marked token is unambiguous: it means both "this number" and
        # "this number as a fraction", and only one of them can be what was meant.
        if candidate.was_percent:
            forms.append(candidate.value / 100)
        elif expected_is_rate:
            forms.append(candidate.value / 100)

        for form in forms:
            if _close(expected, form, tolerance):
                return True
    return False


def _close(expected: float, actual: float, tolerance: float) -> bool:
    if expected == actual:
        return True
    if expected == 0:
        return abs(actual) <= max(tolerance, 1e-9)
    return abs(actual - expected) <= abs(expected) * max(tolerance, 1e-9)


def _contains_label(text: str, label: str) -> bool:
    """Whether a category name appears in the answer.

    Matched on word boundaries so that 'S-05' does not satisfy a question about
    'S-050', and 'North' does not match inside 'Northampton'. Non-word characters in
    the label ('S-07', 'Home & Garden') are escaped rather than stripped.
    """
    pattern = r"(?<![\w-])" + re.escape(label.strip().lower()) + r"(?![\w-])"
    return re.search(pattern, text.lower()) is not None


def _mention_satisfied(text: str, requirement: str) -> bool:
    """One `must_mention` entry, where '|' separates acceptable synonyms.

    Synonyms exist because a requirement like "the answer must say the columns
    disagree" has half a dozen equally correct wordings, and grading on a single
    chosen verb measures vocabulary rather than understanding.
    """
    lowered = text.lower()
    return any(part.strip() in lowered for part in requirement.split("|") if part.strip())


def grade(
    question: Question,
    expected: ExpectedAnswer,
    answer_text: str,
    *,
    tool_calls: int = 0,
    failed_calls: int = 0,
    duration_s: float = 0.0,
    unknown_tools: list[str] | None = None,
) -> Grade:
    """Score one answer. Never raises -- a broken answer is a failing grade, not a crash."""
    result = Grade(
        question_id=question.id,
        category=question.category,
        has_mention_requirement=bool(question.must_mention),
        tool_calls=tool_calls,
        failed_calls=failed_calls,
        duration_s=duration_s,
    )
    text = answer_text or ""
    tolerance = question.tolerance if question.tolerance > 0 else DEFAULT_TOLERANCE

    if unknown_tools:
        result.tools_valid = False
        result.reasons.append(f"called tools that do not exist: {', '.join(unknown_tools)}")

    if not text.strip():
        result.reasons.append("no answer text produced")
        result.values_correct = False
        result.mentions_present = not question.must_mention
        return result

    numbers = extract_numbers(text)

    if question.answer_kind == "assumption_stated":
        # No single right number, so the only thing graded is whether the answer OWNS
        # its interpretation. Requiring an explicit marker matters: the first version
        # passed on keywords alone, which meant an agent that merely listed the column
        # names scored 2/2 here without choosing anything. Naming the column you used
        # is the entire skill this category tests.
        result.values_correct = any(marker in text.lower() for marker in _ASSUMPTION_MARKERS)
        if not result.values_correct:
            result.reasons.append(
                "did not state which interpretation or column the answer is based on"
            )
    elif question.answer_kind == "scalar":
        result.values_correct = _grade_scalar(question, expected, numbers, tolerance, result)
    elif question.answer_kind == "label":
        result.values_correct = _grade_label(expected, text, result)
    elif question.answer_kind == "ranking":
        result.values_correct = _grade_ranking(question, expected, text, result)
    else:
        result.values_correct = _grade_facts(expected, text, numbers, tolerance, result)

    missing = [m for m in question.must_mention if not _mention_satisfied(text, m)]
    result.missing_mentions = missing
    result.mentions_present = not missing
    if missing:
        result.reasons.append(f"did not mention: {', '.join(missing)}")

    return result


def _grade_scalar(
    question: Question,
    expected: ExpectedAnswer,
    numbers: list[ExtractedNumber],
    tolerance: float,
    result: Grade,
) -> bool:
    target = expected.scalar
    if target is None:
        target = expected.rows[0][0]
    if not isinstance(target, (int, float)):
        return _grade_label(expected, "", result)
    if number_matches(float(target), numbers, tolerance):
        return True
    result.missing_values.append(str(target))
    result.reasons.append(f"expected value {target} not found in the answer")
    return False


def _grade_label(expected: ExpectedAnswer, text: str, result: Grade) -> bool:
    label = expected.labels[0] if expected.labels else None
    if label is None:
        result.reasons.append("ground truth has no label to match")
        return False
    if _contains_label(text, label):
        return True
    result.missing_values.append(label)
    result.reasons.append(f"expected answer '{label}' not named")
    return False


def _grade_ranking(question: Question, expected: ExpectedAnswer, text: str, result: Grade) -> bool:
    """The first top_k labels must appear, in the right relative order.

    Position is checked by where each label first occurs in the text. An answer that
    names the right three items in the wrong order has not ranked them, and ranking is
    what the question asked for.
    """
    wanted = expected.labels[: question.top_k or len(expected.labels)]
    lowered = text.lower()
    positions = []
    for label in wanted:
        if not _contains_label(text, str(label)):
            result.missing_values.append(str(label))
            result.reasons.append(f"ranking is missing '{label}'")
            return False
        positions.append(lowered.index(str(label).lower()))

    if positions != sorted(positions):
        result.reasons.append(
            f"ranking order is wrong; expected {' > '.join(str(w) for w in wanted)}"
        )
        return False
    return True


def _grade_facts(
    expected: ExpectedAnswer,
    text: str,
    numbers: list[ExtractedNumber],
    tolerance: float,
    result: Grade,
) -> bool:
    """Every value the reference query produced has to appear somewhere in the answer."""
    ok = True
    for column, value in expected.facts.items():
        for item in value if isinstance(value, list) else [value]:
            if item is None:
                continue
            if isinstance(item, bool):
                continue
            if isinstance(item, (int, float)):
                if not number_matches(float(item), numbers, tolerance):
                    ok = False
                    result.missing_values.append(f"{column}={item}")
            elif isinstance(item, str) and not _contains_label(text, item):
                ok = False
                result.missing_values.append(f"{column}={item}")

    if not ok:
        result.reasons.append(f"missing values: {', '.join(result.missing_values[:6])}")
    return ok


def summarise(grades: list[Grade]) -> dict[str, Any]:
    """Aggregate grades into the numbers a report shows."""
    total = len(grades)
    if total == 0:
        return {"total": 0}

    by_category: dict[str, dict[str, int]] = {}
    for item in grades:
        bucket = by_category.setdefault(item.category, {"total": 0, "correct": 0})
        bucket["total"] += 1
        bucket["correct"] += int(item.correct)

    # Only questions that CARRY a must_mention requirement can pass or fail it.
    # Counting the rest as passes made a stub that mentions nothing score 58% here,
    # which is a statistic about the question set rather than about the agent.
    with_mentions = [g for g in grades if g.has_mention_requirement]

    return {
        "total": total,
        "correct": sum(g.correct for g in grades),
        "values_correct": sum(g.values_correct for g in grades),
        "mentions_total": len(with_mentions),
        "mentions_present": sum(g.mentions_present for g in with_mentions),
        "tool_calls": sum(g.tool_calls for g in grades),
        "failed_calls": sum(g.failed_calls for g in grades),
        "duration_s": round(sum(g.duration_s for g in grades), 2),
        "by_category": dict(sorted(by_category.items())),
    }
