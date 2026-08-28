"""Answering a question that has already been answered, in five milliseconds.

WHY THIS IS THE HIGHEST-VALUE USE OF POSTGRES HERE
--------------------------------------------------
An analysis costs 90-190 seconds of local GPU. The result is a pure function of four
things — dataset, version, question, model — and all four are known before any work
starts. So the second time somebody asks, there is nothing to compute: it is a
primary-key lookup against a unique index.

That also happens to be the honest answer to "you are using Postgres for metadata
only". The queue, the event log and now the cache are all things Postgres is doing that
a metadata store would not.

THE CACHE KEY IS THE CORRECTNESS ARGUMENT
-----------------------------------------
Every part of the key is there to prevent a specific wrong answer:

    dataset_id       obvious
    dataset_version  a new upload must not serve an answer about the old file. Same
                     reason the version is pinned on the analysis in the first place.
    question_hash    normalised, so "Which country earns most?" and "which country
                     earns most" are one entry rather than two
    llm_model        the two models genuinely disagree (60% against 29% on the
                     evaluation set), so a cached Qwen2.5 answer must never be served
                     to somebody who asked for Qwen3

WHAT IS NOT CACHED, AND WHY THAT MATTERS MORE THAN WHAT IS
-----------------------------------------------------------
Failures, and any answer carrying an unverified figure. Replaying a wrong answer
instantly is worse than recomputing it slowly, because **speed reads as confidence**: an
answer that appears immediately looks retrieved and certain, and this system's whole
premise is that a claim is worth what its evidence is worth.

THE NEXT VERSION, DESIGNED FOR AND NOT BUILT
--------------------------------------------
Semantic caching: embed the question, store the vector alongside the hash, and match on
cosine similarity above ~0.8 so "which country earns most" also hits "top country by
revenue". Published implementations report 60-70% hit rates against ~10% for exact
match. It needs the pgvector extension (this deployment runs postgres:16-alpine, which
does not ship it) and an embedding model pulled into Ollama. The exact-match key here is
the subset of that design which costs neither, and the schema has room for the column.

Meanwhile the query rewriter (`app/agent/rewrite.py`) recovers part of the same benefit
for free: it turns "what about France?" into a standalone question, and standalone
questions are what hash consistently.
"""

from __future__ import annotations

import hashlib
import re
import uuid
from typing import Any

from sqlalchemy import select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from app.db.models import AnswerCache

# Trailing punctuation and case carry no meaning in a question, so they must not create
# a second cache entry. Internal punctuation is left alone: "revenue, by country" and
# "revenue by country" are the same to a reader but normalising that way starts down a
# road that ends in stemming, and a wrong cache hit is far worse than a miss.
_PUNCTUATION = re.compile(r"[?!.\s]+$")


def normalise(question: str) -> str:
    return _PUNCTUATION.sub("", " ".join(question.lower().split()))


# Bumped whenever the prompt or the loop changes in a way that would change answers.
#
# WITHOUT THIS THE CACHE OUTLIVES THE CODE. A rule was added telling the model to write
# "84%" rather than "0.8399690286861813"; the very next run served the old answer from
# cache in 552 ms and the fix looked like it had simply not worked. A cache keyed only on
# the question pins yesterday's behaviour to today's build.
PROMPT_VERSION = "2026-08-29.2"


def question_hash(question: str) -> str:
    """The cache key: the normalised question, bound to the prompt that would answer it."""
    # A separator that cannot occur in either part, so no version/question pair can
    # collide with a different one by concatenating to the same string.
    material = f"{PROMPT_VERSION}\n{normalise(question)}"
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def lookup(
    session: Session,
    *,
    dataset_id: uuid.UUID,
    dataset_version: int,
    question: str,
    llm_model: str,
) -> dict[str, Any] | None:
    """A previously computed result, or None.

    The hit counter is incremented here rather than by the caller, so the number is
    evidence about the cache rather than about whoever remembered to record it.
    """
    row = session.scalar(
        select(AnswerCache).where(
            AnswerCache.dataset_id == dataset_id,
            AnswerCache.dataset_version == dataset_version,
            AnswerCache.question_hash == question_hash(question),
            AnswerCache.llm_model == llm_model,
        )
    )
    if row is None:
        return None

    session.execute(
        update(AnswerCache)
        .where(AnswerCache.id == row.id)
        .values(hit_count=AnswerCache.hit_count + 1)
    )

    # Marked so it is never mistaken for fresh work — the UI says so, and the stored
    # analysis says so. An answer that appears in five milliseconds should say why.
    return {**row.result, "cached": True}


def store(
    session: Session,
    *,
    dataset_id: uuid.UUID,
    dataset_version: int,
    question: str,
    llm_model: str,
    result: dict[str, Any],
) -> bool:
    """Remember a result. Returns False when it was deliberately not cached.

    `ON CONFLICT DO NOTHING` rather than check-then-insert: two workers can finish the
    same question at the same moment, and the loser should not turn a completed analysis
    into a unique-violation traceback.
    """
    if not is_cacheable(result):
        return False

    session.execute(
        pg_insert(AnswerCache)
        .values(
            id=uuid.uuid4(),
            dataset_id=dataset_id,
            dataset_version=dataset_version,
            question_hash=question_hash(question),
            question=question.strip(),
            llm_model=llm_model,
            # Stored without the flag, so it is added on the way out rather than baked
            # in — otherwise the first replay would cache "cached: true" forever.
            result={k: v for k, v in result.items() if k != "cached"},
        )
        .on_conflict_do_nothing(
            index_elements=[
                AnswerCache.dataset_id,
                AnswerCache.dataset_version,
                AnswerCache.question_hash,
                AnswerCache.llm_model,
            ]
        )
    )
    return True


def is_cacheable(result: dict[str, Any]) -> bool:
    """Is this a result worth replaying instantly?

    An answer with a warning is exactly the answer that should NOT come back instantly.
    Warnings mean the agent ran out of budget, answered without querying, or wrote a
    figure that could not be traced to a computation — and speed reads as confidence.
    """
    if not result.get("answer"):
        return False
    if result.get("warnings"):
        return False
    # Nothing was computed, so there is nothing worth keeping.
    return bool(result.get("table") or result.get("steps"))
