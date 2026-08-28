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
    "do not do arithmetic yourself"                      53,847 - 47,363 = "16,484"
    "say so when the data cannot answer it"              the plausible-fabrication mode
    "ask for everything in one turn"                     see below

ASKING FOR EVERYTHING AT ONCE IS THE MOST VALUABLE RULE HERE
------------------------------------------------------------
An early version told the model to call ONE tool per turn, on the reasoning that a
small model reasons better about one result at a time. Measured against the clock, that
rule was expensive nonsense:

    one tool call            30-70 milliseconds
    one turn of the model    45-90 seconds

So a turn is roughly a thousand times more expensive than the work it authorises, and
"which country earns most" and "what is the overall total" as two turns costs two
minutes to save a model forty milliseconds of thinking. Told it may batch, qwen3:4b
duly asked for three queries in one 51-second turn.

This is also the answer to "the trace is always two steps": more analysis per question
comes from a wider turn, not from more turns.
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
1. Work out everything that needs computing to answer the question well.
2. Ask for ALL OF IT IN ONE TURN. You may call several tools at once, and you should:
   every tool you name runs before you see any result, each takes milliseconds, and a
   turn of yours costs the person waiting nearly a minute. Four queries in one turn are
   free; four turns of one query each are four minutes of somebody's life.
3. Then write the answer as plain text.

BE GENEROUS ABOUT WHAT YOU ASK FOR, since it is free. A ranking is more useful beside
the total it is a share of; a "highest" is more useful beside the average; a trend is
more useful beside the size of the thing that is trending. If a second query would make
the answer meaningfully better, ask for it in the same turn.

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
- A chart is drawn for you automatically from your final result, and its type is chosen
  from the shape of that result. So think about the shape: a date or month column with
  a value gives a line, a handful of categories with shares gives a pie, two numeric
  columns give a scatter, a ranking gives bars. You do not need to ask for a chart.
- If the data genuinely cannot answer the question, say that plainly and say what the
  data does contain. A wrong answer is far worse than "this dataset does not record
  that".

KEEP YOUR REASONING SHORT
You are given the schema, and every tool result is already in front of you. Do not
re-derive what a result already says, and do not rehearse the answer before writing it.
Reasoning at length about a table you can read costs a user minutes of waiting for
nothing.

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


# The system prompt for the answer turn, replacing the planning one entirely.
#
# The planning prompt is a page of rules about choosing tools, batching calls, avoiding
# identifier columns and repairing failed queries. NONE of it applies once the results
# are in — and a small model does not ignore instructions it cannot use, it reasons
# about them. Measured on the real thing: the answer turn was spending 10,866 characters
# of thinking with the full prompt in front of it.
#
# So the answer turn gets a prompt about writing, and keeps only the two rules that
# still bind: every figure comes from a result, and nothing is invented.
ANSWER_SYSTEM_PROMPT = """\
You are a careful data analyst writing the final answer. The results you need are
in the conversation above and no more work is required.

Write two to four sentences of plain prose. Lead with the direct answer and its figure,
then the one or two comparisons that make it meaningful, then any caveat that matters.

- Every number you write must appear in a result above. Do not calculate anything new,
  including differences, percentages and ratios.
- Do not add a currency symbol or a unit that is not in the data.
- The full results table is displayed to the reader directly beneath your answer, so do
  not list its rows.
- No markdown, no headings, no bullet points, no code blocks.

Do not deliberate. You have the numbers; write the sentences."""

FORCE_ANSWER_PROMPT = (
    "Write the final answer now, in two to four sentences of plain prose, using only "
    "the results above. If those results are not enough to answer the question, say "
    "exactly what you established and what is still missing."
)

NO_EVIDENCE_PROMPT = (
    "You answered without running anything. Every number in an answer must come from a "
    "tool result. Call a tool now to compute what the question actually asks for."
)

EMPTY_TURN_PROMPT = (
    "That turn produced neither a tool call nor an answer. Either call one tool, or "
    "write your final answer in plain prose."
)
