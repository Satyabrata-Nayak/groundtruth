"""What the agent is told about the questions that came before this one.

THE PROBLEM MEMORY SOLVES
-------------------------
Without it, every question is the first question. "Which country earns most?" works;
"what about last year?" is unanswerable, because the worker sees six words and no idea
what is being compared to what. A thread is what makes a follow-up mean something.

WHAT IS CARRIED, AND WHAT IS NOT
--------------------------------
The temptation is to replay the conversation. That is wrong here for a hard reason: the
context window is 8,192 tokens and one tool result is up to 50 rows. Three replayed
turns would evict the schema, and the schema is what stops the model inventing column
names.

So a turn is compressed to the three things a follow-up actually needs:

    the question    so "last year" has something to attach to
    the answer      so "why is that?" has a referent
    THE SQL         so "same thing but for France" is one edit away

The SQL is the part people leave out and it is the most valuable of the three. A model
that can see `SELECT Country, SUM(Quantity*UnitPrice) ... GROUP BY Country` can write
the follow-up by changing one clause. A model given only the prose has to rediscover the
whole query from the schema.

Tool payloads, event trails and reasoning traces are all excluded. They are large, and
none of them is something a follow-up refers to.

WHY THREE TURNS
---------------
The literature converges on 3-5 for a rolling window, and the binding constraint here is
smaller than the literature's: at ~120 tokens per compressed turn, three costs ~360 and
five costs ~600 out of a budget where the schema and samples already take ~1,200.
Three keeps the recent context that follow-ups actually reach for and leaves room for
the results.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import Analysis, AnalysisStatus

# How many previous exchanges to carry. See the note above.
MAX_TURNS = 3
# A long answer is truncated rather than dropped: the first sentence carries the
# referent ("the United Kingdom"), which is the part a follow-up points at.
_MAX_ANSWER_CHARS = 320
_MAX_SQL_CHARS = 280


@dataclass(frozen=True)
class Turn:
    """One previous exchange, already compressed to what a follow-up needs."""

    question: str
    answer: str
    sql: str | None


def recent_turns(
    session: Session, conversation_id: uuid.UUID | None, *, limit: int = MAX_TURNS
) -> list[Turn]:
    """The last `limit` successful exchanges in a thread, oldest first.

    Only SUCCEEDED analyses are carried. A failed one has no answer to refer back to,
    and a cancelled one was abandoned on purpose — replaying either as context would be
    telling the model that something happened which did not.
    """
    if conversation_id is None:
        return []

    rows = session.scalars(
        select(Analysis)
        .where(
            Analysis.conversation_id == conversation_id,
            Analysis.status == AnalysisStatus.SUCCEEDED,
        )
        # Newest first with a LIMIT, then reversed — the alternative reads the whole
        # thread to take the tail of it.
        .order_by(Analysis.turn_index.desc().nullslast(), Analysis.created_at.desc())
        .limit(limit)
    ).all()

    return [turn for row in reversed(rows) if (turn := _compress(row)) is not None]


def _compress(analysis: Analysis) -> Turn | None:
    result = analysis.result or {}
    answer = str(result.get("answer") or "").strip()
    if not answer:
        return None
    return Turn(
        question=analysis.question.strip(),
        answer=_clip(answer, _MAX_ANSWER_CHARS),
        sql=_last_sql(result),
    )


def _last_sql(result: dict) -> str | None:
    """The SQL that produced the answer, which is the query a follow-up edits."""
    for step in reversed(result.get("steps") or []):
        sql = (step.get("arguments") or {}).get("sql")
        if step.get("ok") and sql:
            return _clip(" ".join(str(sql).split()), _MAX_SQL_CHARS)
    return None


def _clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def render(turns: list[Turn]) -> str:
    """The turns as a prompt block, or an empty string when there is no history.

    Returning "" rather than "no previous questions" is deliberate: a block that says
    nothing still costs tokens and still invites the model to reason about why it is
    empty.
    """
    if not turns:
        return ""

    blocks = []
    for index, turn in enumerate(turns, start=1):
        lines = [f"  {index}. asked: {turn.question}", f"     answered: {turn.answer}"]
        if turn.sql:
            lines.append(f"     using: {turn.sql}")
        blocks.append("\n".join(lines))

    return (
        "EARLIER IN THIS CONVERSATION\n"
        "Use these only to understand what the new question refers to. Do NOT reuse an "
        "old number as an answer — recompute anything you are asked about.\n\n" + "\n".join(blocks)
    )
