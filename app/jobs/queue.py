"""A job queue built out of one Postgres table.

    POST /analyses ──► INSERT a row (PENDING)          ← the API's whole job
                              │
                              │   ...time passes, possibly a restart...
                              ▼
    worker loop    ──► UPDATE ... FOR UPDATE SKIP LOCKED  (PENDING → RUNNING)
                              │
                              ├── heartbeat every 5s     (proof of life)
                              │
                              ▼
                       UPDATE ... (RUNNING → SUCCEEDED | FAILED | CANCELLED)

THE RACE THIS EXISTS TO PREVENT
-------------------------------
Two workers poll at the same instant. Both run `SELECT ... WHERE status='PENDING'
LIMIT 1`, both see analysis #7, both mark it RUNNING, both analyse it. The user's
question is answered twice and, in M5, the model is invoked twice for one answer.

The naive fixes are all worse than they look:

    a global lock         the queue becomes single-file; workers stop being parallel
    LIMIT 1 + UPDATE      still two statements; another transaction can interleave
    optimistic retry      works, but every worker does wasted work under contention
    SELECT ... FOR UPDATE correct, but worker B *blocks* on A's lock, and blocking on
                          a row you do not want is the definition of a queue that
                          does not scale

`FOR UPDATE SKIP LOCKED` is the one that fits: it takes a row lock and, instead of
waiting for rows another transaction already locked, walks straight past them. Two
workers polling simultaneously get two different jobs, and neither waits.

WHY THE CLAIM IS A SINGLE STATEMENT
-----------------------------------
`UPDATE ... WHERE id = (SELECT ... FOR UPDATE SKIP LOCKED LIMIT 1) RETURNING *`

Selecting and then updating in two statements leaves a window between them. One
statement has no window: the row is located, locked and mutated atomically, and
RETURNING hands back what was written. This is the shape to copy; the two-statement
version is subtly wrong in a way tests rarely catch.

WHY EVERY WRITE IS GUARDED BY worker_id
---------------------------------------
This is the part that makes crash recovery safe, and it is easy to leave out.

    12:00:00  worker A claims #7
    12:00:31  A is paused (long GC, suspended laptop) and misses its heartbeats
    12:00:35  the sweep sees a stale heartbeat, requeues #7
    12:00:36  worker B claims #7 and starts over
    12:00:40  A wakes up, finishes, and writes its result

Without a guard, A's write lands on a row B now owns, and B's write lands after it.
So every terminal write carries `AND worker_id = :me AND status = 'RUNNING'`. A's
UPDATE matches zero rows and is silently a no-op — which is exactly right, because
A was, as far as the system is concerned, dead.

TIME COMES FROM THE DATABASE, NEVER FROM PYTHON
-----------------------------------------------
Every timestamp here is `now()` evaluated by Postgres. If the worker stamped
heartbeats from its own clock and the sweep compared them against another machine's
clock, a few seconds of skew would either resurrect live jobs or never reclaim dead
ones. One clock, and it belongs to the database.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import timedelta
from typing import Any

from sqlalchemy import Select, func, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from app.db.models import Analysis, AnalysisEvent, AnalysisStatus, EventKind
from app.jobs import notify


@dataclass(frozen=True)
class ClaimedAnalysis:
    """A snapshot of a claimed job, detached from the session that claimed it.

    Plain values rather than an ORM instance, because the worker runs the analysis
    outside the claiming transaction. Handing it a live ORM object would invite a
    lazy load on a closed session — the classic "DetachedInstanceError" — and, worse,
    would let the worker read a field it thinks is fresh and is actually minutes old.
    """

    id: uuid.UUID
    dataset_id: uuid.UUID
    dataset_version: int
    question: str
    attempts: int
    worker_id: str
    # What the asker chose, if anything. None for either means "use the worker's
    # configuration", which is what every row written before the columns existed says.
    llm_model: str | None = None
    llm_thinking: bool | None = None
    # The thread this question belongs to, so the worker can load its history.
    conversation_id: uuid.UUID | None = None


@dataclass(frozen=True)
class HeartbeatOutcome:
    """What one heartbeat learned. Three questions answered by one round trip.

    `still_owned` false means the row was reclaimed or finished by somebody else, and
    this worker must stop: anything it computes from here on is unwanted.
    """

    still_owned: bool
    cancel_requested: bool


@dataclass(frozen=True)
class ReclaimReport:
    requeued: list[uuid.UUID]
    abandoned: list[uuid.UUID]

    def __bool__(self) -> bool:
        return bool(self.requeued or self.abandoned)


# --------------------------------------------------------------------------------
# Producing work
# --------------------------------------------------------------------------------


def enqueue(
    session: Session,
    *,
    dataset_id: uuid.UUID,
    dataset_version: int,
    question: str,
    idempotency_key: str | None = None,
    llm_model: str | None = None,
    llm_thinking: bool | None = None,
    conversation_id: uuid.UUID | None = None,
    turn_index: int | None = None,
) -> tuple[Analysis, bool]:
    """Add a question to the queue. Returns (analysis, created).

    `created` is False when an `idempotency_key` matched an existing row, in which
    case that row is returned untouched and nothing is enqueued.

    The insert is `ON CONFLICT DO NOTHING` rather than "check, then insert". The
    check-first version has a window: two retries of the same request can both find
    nothing and both insert, and only one survives the unique index — as a 500. Here
    the database decides, atomically, and the loser simply reads what the winner
    wrote.
    """
    if idempotency_key is None:
        analysis = Analysis(
            dataset_id=dataset_id,
            dataset_version=dataset_version,
            question=question,
            status=AnalysisStatus.PENDING,
            llm_model=llm_model,
            llm_thinking=llm_thinking,
            conversation_id=conversation_id,
            turn_index=turn_index,
        )
        session.add(analysis)
        session.flush()
        emit(session, analysis.id, EventKind.QUEUED, "queued")
        # Wakes an idle worker the moment this commits, instead of leaving it to
        # notice on its next poll. Best-effort: the poll is still the guarantee.
        notify.notify_new_work(session)
        return analysis, True

    stmt = (
        pg_insert(Analysis)
        .values(
            id=uuid.uuid4(),
            dataset_id=dataset_id,
            dataset_version=dataset_version,
            question=question,
            idempotency_key=idempotency_key,
            status=AnalysisStatus.PENDING,
            llm_model=llm_model,
            llm_thinking=llm_thinking,
            conversation_id=conversation_id,
            turn_index=turn_index,
        )
        .on_conflict_do_nothing(index_elements=[Analysis.idempotency_key])
        .returning(Analysis.id)
    )
    inserted_id = session.execute(stmt).scalar_one_or_none()

    if inserted_id is None:
        existing = session.scalars(
            select(Analysis).where(Analysis.idempotency_key == idempotency_key)
        ).one()
        return existing, False

    notify.notify_new_work(session)
    analysis = session.get(Analysis, inserted_id)
    assert analysis is not None  # just inserted in this transaction
    emit(session, analysis.id, EventKind.QUEUED, "queued")
    return analysis, True


# --------------------------------------------------------------------------------
# Consuming work
# --------------------------------------------------------------------------------


def _next_pending_id() -> Select[tuple[uuid.UUID]]:
    """The subquery that picks and locks exactly one waiting job.

    FIFO by `created_at`. `skip_locked=True` is what makes several workers useful:
    without it, the second worker blocks on the first worker's row lock and the queue
    becomes single-file.
    """
    return (
        select(Analysis.id)
        .where(Analysis.status == AnalysisStatus.PENDING)
        .order_by(Analysis.created_at)
        .limit(1)
        .with_for_update(skip_locked=True)
    )


def claim_next(session: Session, worker_id: str) -> ClaimedAnalysis | None:
    """Take ownership of the oldest waiting analysis, or return None if there is none.

    `attempts` is incremented HERE, at claim time, not when the work finishes. A job
    that crashes its worker has still consumed an attempt, which is the only thing
    that stops a poison job from being retried until the end of time.
    """
    stmt = (
        update(Analysis)
        .where(Analysis.id == _next_pending_id().scalar_subquery())
        .values(
            status=AnalysisStatus.RUNNING,
            worker_id=worker_id,
            attempts=Analysis.attempts + 1,
            started_at=func.now(),
            heartbeat_at=func.now(),
            cancel_requested=False,
        )
        .returning(
            Analysis.id,
            Analysis.dataset_id,
            Analysis.dataset_version,
            Analysis.question,
            Analysis.attempts,
            # The asker's model choice travels with the claim. Stored and then not
            # returned would be the worst of both worlds: the row would claim an answer
            # came from a model that never saw the question.
            Analysis.llm_model,
            Analysis.llm_thinking,
            Analysis.conversation_id,
        )
        .execution_options(synchronize_session=False)
    )
    row = session.execute(stmt).one_or_none()
    if row is None:
        return None

    claimed = ClaimedAnalysis(
        id=row.id,
        dataset_id=row.dataset_id,
        dataset_version=row.dataset_version,
        question=row.question,
        attempts=row.attempts,
        worker_id=worker_id,
        llm_model=row.llm_model,
        llm_thinking=row.llm_thinking,
        conversation_id=row.conversation_id,
    )
    emit(
        session,
        claimed.id,
        EventKind.CLAIMED,
        f"claimed by {worker_id} (attempt {claimed.attempts})",
    )
    return claimed


def heartbeat(session: Session, analysis_id: uuid.UUID, worker_id: str) -> HeartbeatOutcome:
    """Say "still alive", and find out whether anyone still wants the answer.

    One statement does three jobs: it refreshes the liveness timestamp, it proves
    ownership (the guard clause), and it reads back the cancellation flag. Splitting
    these into three queries would triple the round trips a long analysis makes for
    no gain — and would open a window where a worker is told it still owns a row that
    was reclaimed between the two reads.
    """
    stmt = (
        update(Analysis)
        .where(
            Analysis.id == analysis_id,
            Analysis.worker_id == worker_id,
            Analysis.status == AnalysisStatus.RUNNING,
        )
        .values(heartbeat_at=func.now())
        .returning(Analysis.cancel_requested)
        .execution_options(synchronize_session=False)
    )
    row = session.execute(stmt).one_or_none()
    if row is None:
        return HeartbeatOutcome(still_owned=False, cancel_requested=False)
    return HeartbeatOutcome(still_owned=True, cancel_requested=bool(row.cancel_requested))


def reclaim_stalled(session: Session, *, timeout_s: float, max_attempts: int) -> ReclaimReport:
    """Recover analyses whose worker stopped heartbeating.

    Two disjoint groups, and the distinction matters:

        attempts >= max_attempts  ->  FAILED     it has already killed N workers;
                                                 requeueing it just kills another
        attempts <  max_attempts  ->  PENDING    a plain crash, worth another go

    Requeueing clears `worker_id` and `heartbeat_at`, so nothing about the dead
    worker's ownership survives into the next attempt.

    Safe to run from every worker concurrently. Two sweeps racing on the same row
    serialise on its lock, and under READ COMMITTED the second one re-evaluates its
    WHERE clause against the row the first one just wrote — which no longer matches,
    so it updates nothing.
    """
    cutoff = func.now() - timedelta(seconds=timeout_s)
    stale = (
        Analysis.status == AnalysisStatus.RUNNING,
        Analysis.heartbeat_at.is_not(None),
        Analysis.heartbeat_at < cutoff,
    )

    abandoned = list(
        session.scalars(
            update(Analysis)
            .where(*stale, Analysis.attempts >= max_attempts)
            .values(
                status=AnalysisStatus.FAILED,
                finished_at=func.now(),
                error=(
                    f"abandoned after {max_attempts} attempts: the worker stopped "
                    f"reporting in each time"
                ),
                worker_id=None,
                heartbeat_at=None,
            )
            .returning(Analysis.id)
            .execution_options(synchronize_session=False)
        )
    )

    requeued = list(
        session.scalars(
            update(Analysis)
            .where(*stale, Analysis.attempts < max_attempts)
            .values(
                status=AnalysisStatus.PENDING,
                worker_id=None,
                heartbeat_at=None,
                started_at=None,
            )
            .returning(Analysis.id)
            .execution_options(synchronize_session=False)
        )
    )

    for analysis_id in abandoned:
        emit(
            session,
            analysis_id,
            EventKind.FAILED,
            f"abandoned after {max_attempts} attempts without a heartbeat",
        )
    for analysis_id in requeued:
        emit(
            session,
            analysis_id,
            EventKind.RECLAIMED,
            "worker stopped reporting in; requeued for another attempt",
        )

    return ReclaimReport(requeued=requeued, abandoned=abandoned)


# --------------------------------------------------------------------------------
# Finishing work
# --------------------------------------------------------------------------------


def _finish(
    session: Session,
    analysis_id: uuid.UUID,
    worker_id: str,
    *,
    status: AnalysisStatus,
    result: dict[str, Any] | None = None,
    error: str | None = None,
) -> bool:
    """Move a job to a terminal state, but only if this worker still owns it.

    Returns False when the guard rejected the write — the row was reclaimed while
    this worker was working. The caller must treat that as "my result is unwanted"
    and discard it, not retry.
    """
    stmt = (
        update(Analysis)
        .where(
            Analysis.id == analysis_id,
            Analysis.worker_id == worker_id,
            Analysis.status == AnalysisStatus.RUNNING,
        )
        .values(
            status=status,
            result=result,
            error=error,
            finished_at=func.now(),
            heartbeat_at=None,
        )
        .returning(Analysis.id)
        .execution_options(synchronize_session=False)
    )
    return session.execute(stmt).one_or_none() is not None


def succeed(
    session: Session, analysis_id: uuid.UUID, worker_id: str, result: dict[str, Any]
) -> bool:
    if not _finish(session, analysis_id, worker_id, status=AnalysisStatus.SUCCEEDED, result=result):
        return False
    emit(session, analysis_id, EventKind.SUCCEEDED, "analysis complete")
    return True


def fail(session: Session, analysis_id: uuid.UUID, worker_id: str, error: str) -> bool:
    if not _finish(session, analysis_id, worker_id, status=AnalysisStatus.FAILED, error=error):
        return False
    emit(session, analysis_id, EventKind.FAILED, error[:500])
    return True


def mark_cancelled(session: Session, analysis_id: uuid.UUID, worker_id: str) -> bool:
    """The worker's side of a cancellation: it noticed the flag and stopped."""
    if not _finish(
        session,
        analysis_id,
        worker_id,
        status=AnalysisStatus.CANCELLED,
        error="cancelled by request",
    ):
        return False
    emit(session, analysis_id, EventKind.CANCELLED, "stopped at the caller's request")
    return True


def cancel(session: Session, analysis_id: uuid.UUID) -> str | None:
    """The caller's side of a cancellation. Returns the resulting status, or None.

    A PENDING job is cancelled outright — nobody is running it, so there is nothing
    to negotiate with. A RUNNING job only gets a *flag*: the worker owns the DuckDB
    connection, the open files and (in M5) the model call, and it is the only party
    that can stop cleanly and record where it stopped. It sees the flag at its next
    heartbeat, within one beat interval.

    Cancelling an already-finished job does nothing and says so, rather than
    pretending to succeed.
    """
    analysis = session.get(Analysis, analysis_id, with_for_update=True)
    if analysis is None:
        return None

    if analysis.status == AnalysisStatus.PENDING:
        analysis.status = AnalysisStatus.CANCELLED
        analysis.cancel_requested = True
        analysis.finished_at = func.now()
        analysis.error = "cancelled before it started"
        emit(session, analysis_id, EventKind.CANCELLED, "cancelled while still queued")
        return AnalysisStatus.CANCELLED

    if analysis.status == AnalysisStatus.RUNNING:
        analysis.cancel_requested = True
        emit(
            session,
            analysis_id,
            EventKind.NOTE,
            "cancellation requested; waiting for the worker to stop",
        )
        return AnalysisStatus.RUNNING

    return analysis.status


# --------------------------------------------------------------------------------
# The trail
# --------------------------------------------------------------------------------


def emit(
    session: Session,
    analysis_id: uuid.UUID,
    kind: EventKind,
    message: str,
    payload: dict[str, Any] | None = None,
) -> AnalysisEvent:
    """Record one observable thing that happened.

    Written in the SAME transaction as the state change it describes. That is the
    whole discipline: if the status change commits, its event committed with it, so
    the trail can never claim something the row disagrees with.
    """
    event = AnalysisEvent(analysis_id=analysis_id, kind=kind, message=message, payload=payload)
    session.add(event)
    return event
