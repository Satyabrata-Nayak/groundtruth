"""What the model is told before it is asked anything.

THE SCHEMA IS HANDED OVER, NOT DISCOVERED
-----------------------------------------
The obvious agent design starts with an empty prompt and lets the model call
`inspect_schema` as its first move. That is one full model turn — on a 4B model with
reasoning on, 10 to 40 seconds — spent producing a result that is completely
deterministic and that we can fetch in 40 milliseconds without asking anyone.

Worse, it is the turn most likely to go wrong. A model that has not seen the column
names has nothing to ground its first tool call in, so it guesses names, gets an error
back, and spends a second turn recovering. Handing the schema over removes an entire
class of failure and roughly a third of the wall-clock time.

`inspect_schema` and `profile_column` stay in the action space regardless: the summary
here is deliberately compact, and a model that wants a column's frequent values or its
exact null count should be able to go and get them.

SAMPLE ROWS ARE WORTH THEIR TOKENS
----------------------------------
Three real rows cost ~150 tokens and answer questions a type list cannot: is
`InvoiceDate` an ISO timestamp or `1/10/11 10:04`? Is `Country` 'UK' or 'United
Kingdom'? Does `Quantity` go negative? A model writing `WHERE Country = 'UK'` against
a column that says 'United Kingdom' produces an empty result and a confident answer of
zero, which is the exact failure this project exists to prevent.

THE RULES ARE ABOUT GROUNDING, NOT POLITENESS
---------------------------------------------
Every instruction in the system prompt earns its place by preventing a specific
observed failure:

    "never state a number you did not get from a tool"   the whole point of the system
    "derived metrics need execute_sql"                   compare_groups cannot express
                                                         Quantity * UnitPrice
    "check the schema before naming a column"            invented column names
    "one tool call per turn"                             qwen3 emits parallel calls it
                                                         cannot reason about together
    "say so when the data cannot answer it"              the plausible-fabrication mode
"""

from __future__ import annotations

from typing import Any

from app.data.sandbox import TABLE_NAME

# How many sample rows to show, and how wide a single value may be before it is cut.
# Three rows is enough to reveal a format; a 4,000 character JSON blob in one cell is
# not worth the context it costs.
_SAMPLE_ROWS = 3
_MAX_VALUE_CHARS = 60

SYSTEM_PROMPT = f"""\
You are a careful data analyst. You answer questions about ONE dataset by running \
tools against it, and you never answer from memory or intuition.

The dataset is a single SQL table named "{TABLE_NAME}". You are told its schema below.

HOW TO WORK
1. Decide what needs to be computed to answer the question.
2. Call ONE tool per turn. Read its result before deciding the next step.
3. When you have the numbers, stop calling tools and write the answer as plain text.

RULES YOU MUST FOLLOW
- Every number, name, date and ranking in your answer must come from a tool result in
  this conversation. If you did not compute it, you do not know it.
- Use exact column names from the schema. If a column you want does not exist, look at
  what does exist and use that, or say the question cannot be answered from this data.
- A metric that is not a stored column - revenue as quantity * price, a ratio, a
  month extracted from a date - must be computed with execute_sql. compare_groups can
  only aggregate a column that already exists.
- Aggregate in SQL. Never ask for many raw rows and add them up yourself; you will get
  it wrong and you will only be shown the first 50 rows anyway.
- When the question asks which item is highest, lowest or best, return the TOP 10 with
  their values, not LIMIT 1. The winner is only meaningful next to the runners-up, and
  a single row shows nobody whether it won by a mile or by a rounding error.
- Do NOT do arithmetic yourself. Differences, ratios, percentages and growth rates
  must be computed by SQL, not in your head. If you want "how much bigger", add it to
  the query. Quoting two figures the database produced and letting the reader compare
  them is always correct; subtracting them yourself is how a wrong number gets into a
  sentence where everything else is right.
- Do not add a currency symbol, a unit or a label that is not in the data. If the
  column is called UnitPrice and nothing says which currency, the number has no
  currency. Inventing one is inventing a fact.
- If a tool returns an error, read it: it names the valid columns or the correct
  argument. Fix the call rather than repeating it.
- If the data genuinely cannot answer the question, say that plainly and say what the
  data does contain. A wrong answer is far worse than "this dataset does not record
  that".

YOUR FINAL ANSWER
Two to four sentences of plain prose. Lead with the direct answer and its figure, then
the one or two comparisons that make it meaningful, then any caveat that matters
(nulls you excluded, a filter you applied, a tie).

The table of results is shown to the reader directly beneath your answer, so DO NOT
list every row in prose. Naming the top result and how it compares to the next one is
the whole job; reciting ten rows the reader can already see is slower to produce and
worse to read. No markdown headings, no bullet lists, no code blocks."""


def build_system_prompt(schema: dict[str, Any], samples: dict[str, Any] | None) -> str:
    """The full system message: the standing rules plus this dataset's shape."""
    parts = [SYSTEM_PROMPT, "", "DATASET SCHEMA", render_schema(schema)]
    rendered_samples = render_samples(samples)
    if rendered_samples:
        parts += ["", "SAMPLE ROWS", rendered_samples]
    return "\n".join(parts)


def render_schema(schema: dict[str, Any]) -> str:
    """A compact, fixed-width rendering of `inspect_schema`'s payload.

    A table rather than the raw JSON: the JSON is roughly twice the tokens for the same
    facts, and every one of those tokens is repeated on every turn of the conversation.
    """
    columns: list[dict[str, Any]] = schema.get("columns", [])
    row_count = schema.get("row_count")

    header = f'Table "{TABLE_NAME}"'
    if row_count is not None:
        header += f", {row_count:,} rows, {len(columns)} columns"

    lines = [header, ""]
    name_width = max((len(str(c.get("name", ""))) for c in columns), default=4)
    type_width = max((len(str(c.get("type", ""))) for c in columns), default=4)

    for column in columns:
        name = str(column.get("name", ""))
        parts = [
            f"  {name:<{name_width}}  {str(column.get('type', '')):<{type_width}}",
            f"  {column.get('kind', '')}",
        ]
        distinct = column.get("distinct_count")
        if distinct is not None:
            parts.append(f", {distinct:,} distinct")
        nulls = column.get("null_fraction")
        if nulls:
            parts.append(f", {nulls * 100:.0f}% null")
        warning = column.get("warning")
        if warning:
            # Only the short form: the model needs the verdict, not the essay.
            parts.append(f"  [{warning.split(':')[0]}]")
        lines.append("".join(parts))

    return "\n".join(lines)


def render_samples(samples: dict[str, Any] | None) -> str:
    """The first few real rows, as `column = value` lines.

    Column-per-line rather than a CSV-style grid because a grid with eight wide columns
    wraps in the model's context and stops being alignable, while `Country = United
    Kingdom` survives any wrapping.
    """
    if not samples:
        return ""
    columns = samples.get("columns") or []
    rows = samples.get("rows") or []
    if not columns or not rows:
        return ""

    blocks = []
    for index, row in enumerate(rows[:_SAMPLE_ROWS], start=1):
        pairs = []
        for column, value in zip(columns, row, strict=False):
            text = "NULL" if value is None else str(value)
            if len(text) > _MAX_VALUE_CHARS:
                text = text[: _MAX_VALUE_CHARS - 1] + "…"
            pairs.append(f"{column} = {text}")
        blocks.append(f"  row {index}: " + " | ".join(pairs))
    return "\n".join(blocks)


def build_user_prompt(question: str) -> str:
    """The question, with the one framing instruction that belongs next to it."""
    return (
        f"{question.strip()}\n\n"
        f"Compute the answer from the dataset using the tools, then state it in plain "
        f"prose."
    )


FORCE_ANSWER_PROMPT = (
    "You have used your tool budget. Do not call any more tools. Write your final "
    "answer now using only the results already in this conversation. If those results "
    "are not enough to answer the question, say exactly what you established and what "
    "is still missing."
)

NO_EVIDENCE_PROMPT = (
    "You answered without running anything. Every number in an answer must come from a "
    "tool result. Call a tool now to compute what the question actually asks for."
)

EMPTY_TURN_PROMPT = (
    "That turn produced neither a tool call nor an answer. Either call one tool, or "
    "write your final answer in plain prose."
)
