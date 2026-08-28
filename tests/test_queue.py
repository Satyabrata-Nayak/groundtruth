"""The job queue: claiming, heartbeats, cancellation and crash recovery.

WHAT THESE TESTS ARE ACTUALLY FOR
---------------------------------
Almost every assertion here is about a RACE, and races do not fail loudly. A queue
that hands the same job to two workers works perfectly in development, works in CI,
and produces two charges on a customer's card the first time two workers happen to
poll in the same millisecond. So the important tests below open two real database
sessions and interleave them by hand, because a single-session test cannot observe
the bug it is supposed to prevent.

Everything here needs real Postgres: `SELECT ... FOR UPDATE SKIP LOCKED`, partial
indexes, `ON CONFLICT DO NOTHING` and row-level locking are the things under test, and
none of them exist in SQLite. Testing this against a fake would test the fake.
"""

from __future__ import annotations

import uuid

import pytest
from sqlalchemy import select

from app.db.models import Analysis, AnalysisEvent, AnalysisStatus, Dataset, EventKind
from app.db.session import get_sessionmaker, session_scope
from app.jobs import queue

pytestmark = pytest.mark.integration


@pytest.fixture
def dataset_id(db):
    """A bare dataset row to hang analyses off. No files, no profile — the queue does
    not care, and creating a real dataset would make every test in this file depend on
    ingestion working."""
    with session_scope() as session:
        dataset = Dataset(id=uuid.uuid4(), name="queue-fixture")
        session.add(dataset)
        session.flush()
        return dataset.id


def add(dataset_id: uuid.UUID, question: str = "q", **kwargs) -> uuid.UUID:
    with session_scope() as session:
        analysis, _ = queue.enqueue(
            session, dataset_id=dataset_id, dataset_version=1, question=question, **kwargs
        )
        return analysis.id


def load(analysis_id: uuid.UUID) -> Analysis:
    with session_scope() as session:
        analysis = session.get(Analysis, analysis_id)
        assert analysis is not None
        return analysis


def kinds(analysis_id: uuid.UUID) -> list[str]:
    with session_scope() as session:
        return [
            event.kind
            for event in session.scalars(
                select(AnalysisEvent)
                .where(AnalysisEvent.analysis_id == analysis_id)
                .order_by(AnalysisEvent.id)
            )
        ]


# ------------------------------------------------------------------ enqueueing


def test_enqueue_creates_a_pending_row_and_an_event(dataset_id):
    analysis_id = add(dataset_id, "why did revenue fall?")
    analysis = load(analysis_id)

    assert analysis.status == AnalysisStatus.PENDING
    assert analysis.attempts == 0
    assert analysis.worker_id is None
    assert analysis.started_at is None
    assert kinds(analysis_id) == [EventKind.QUEUED]


def test_an_idempotency_key_makes_the_second_enqueue_a_no_op(dataset_id):
    """The point of the key: a retried request must not become a second job."""
    first = add(dataset_id, "first", idempotency_key="key-1")

    with session_scope() as session:
        analysis, created = queue.enqueue(
            session,
            dataset_id=dataset_id,
            dataset_version=1,
            question="a completely different question",
            idempotency_key="key-1",
        )
        assert created is False
        assert analysis.id == first
        # The stored question is the FIRST one. The second request is not a partial
        # update of the first; it is a retry of it, and rewriting the row would let a
        # retry silently change what is being analysed.
        assert analysis.question == "first"

    assert kinds(first) == [EventKind.QUEUED]  # not queued twice


def test_different_keys_create_different_jobs(dataset_id):
    assert add(dataset_id, idempotency_key="a") != add(dataset_id, idempotency_key="b")


def test_no_key_means_no_deduplication(dataset_id):
    """Two identical questions with no key are two genuine requests, not a mistake."""
    assert add(dataset_id, "same") != add(dataset_id, "same")


# --------------------------------------------------------------------- claiming


def test_claiming_an_empty_queue_returns_none(db, dataset_id):
    with session_scope() as session:
        assert queue.claim_next(session, "worker-1") is None


def test_claim_marks_the_row_and_counts_the_attempt(dataset_id):
    analysis_id = add(dataset_id)
    with session_scope() as session:
        claimed = queue.claim_next(session, "worker-1")

    assert claimed is not None
    assert claimed.id == analysis_id
    analysis = load(analysis_id)
    assert analysis.status == AnalysisStatus.RUNNING
    assert analysis.worker_id == "worker-1"
    assert analysis.attempts == 1
    assert analysis.started_at is not None
    assert analysis.heartbeat_at is not None


def test_claiming_is_fifo(dataset_id):
    first = add(dataset_id, "first")
    second = add(dataset_id, "second")

    with session_scope() as session:
        assert queue.claim_next(session, "w").id == first
    with session_scope() as session:
        assert queue.claim_next(session, "w").id == second


def test_two_workers_claiming_at_once_get_different_jobs(dataset_id):
    """THE test in this file.

    Two transactions are opened and both claim BEFORE either commits — the exact
    interleaving that a sequential test can never produce. Without SKIP LOCKED the
    second worker would block on the first worker's row lock until it committed and
    then take the same job; without the single-statement claim, both would read the
    same PENDING row and both would mark it RUNNING.
    """
    first = add(dataset_id, "first")
    second = add(dataset_id, "second")

    Session = get_sessionmaker()
    session_a, session_b = Session(), Session()
    try:
        claimed_a = queue.claim_next(session_a, "worker-a")
        claimed_b = queue.claim_next(session_b, "worker-b")

        assert claimed_a is not None and claimed_b is not None
        assert {claimed_a.id, claimed_b.id} == {first, second}
        assert claimed_a.id != claimed_b.id

        session_a.commit()
        session_b.commit()
    finally:
        session_a.close()
        session_b.close()


def test_a_third_worker_finds_nothing_while_two_jobs_are_held(dataset_id):
    """SKIP LOCKED skips; it does not wait. A worker with no work must learn that
    immediately, not block until somebody else commits."""
    add(dataset_id, "first")
    add(dataset_id, "second")

    Session = get_sessionmaker()
    session_a, session_b, session_c = Session(), Session(), Session()
    try:
        assert queue.claim_next(session_a, "a") is not None
        assert queue.claim_next(session_b, "b") is not None
        assert queue.claim_next(session_c, "c") is None
        session_a.commit()
        session_b.commit()
    finally:
        for session in (session_a, session_b, session_c):
            session.close()


# ------------------------------------------------------------------- heartbeats


def test_heartbeat_from_the_owner_is_accepted(dataset_id):
    add(dataset_id)
    with session_scope() as session:
        claimed = queue.claim_next(session, "worker-1")
    with session_scope() as session:
        outcome = queue.heartbeat(session, claimed.id, "worker-1")
    assert outcome.still_owned is True
    assert outcome.cancel_requested is False


def test_heartbeat_from_a_stranger_is_refused(dataset_id):
    """The ownership guard. A worker that was reclaimed must find out, and the way it
    finds out is that its heartbeat stops being accepted."""
    add(dataset_id)
    with session_scope() as session:
        claimed = queue.claim_next(session, "worker-1")
    with session_scope() as session:
        assert queue.heartbeat(session, claimed.id, "worker-2").still_owned is False


def test_heartbeat_on_a_finished_analysis_is_refused(dataset_id):
    add(dataset_id)
    with session_scope() as session:
        claimed = queue.claim_next(session, "worker-1")
    with session_scope() as session:
        queue.succeed(session, claimed.id, "worker-1", {"answer": "done"})
    with session_scope() as session:
        assert queue.heartbeat(session, claimed.id, "worker-1").still_owned is False


# ----------------------------------------------------------------- finishing


def test_succeed_stores_the_result(dataset_id):
    add(dataset_id)
    with session_scope() as session:
        claimed = queue.claim_next(session, "worker-1")
    with session_scope() as session:
        assert queue.succeed(session, claimed.id, "worker-1", {"answer": "42"}) is True

    analysis = load(claimed.id)
    assert analysis.status == AnalysisStatus.SUCCEEDED
    assert analysis.result == {"answer": "42"}
    assert analysis.finished_at is not None
    assert analysis.heartbeat_at is None
    assert EventKind.SUCCEEDED in kinds(claimed.id)


def test_a_worker_that_no_longer_owns_a_job_cannot_write_its_result(dataset_id):
    """The scenario the guard exists for.

    Worker A is slow, gets reclaimed, worker B takes over. A then finishes and tries
    to write. Its write must land on nothing — and B's result must survive.
    """
    add(dataset_id)
    with session_scope() as session:
        claimed = queue.claim_next(session, "worker-a")
    with session_scope() as session:
        queue.reclaim_stalled(session, timeout_s=0, max_attempts=5)
    with session_scope() as session:
        requeued = queue.claim_next(session, "worker-b")
    assert requeued is not None and requeued.id == claimed.id

    with session_scope() as session:
        assert queue.succeed(session, claimed.id, "worker-a", {"answer": "stale"}) is False
    with session_scope() as session:
        assert queue.succeed(session, claimed.id, "worker-b", {"answer": "fresh"}) is True

    assert load(claimed.id).result == {"answer": "fresh"}


def test_a_terminal_analysis_cannot_be_written_twice(dataset_id):
    add(dataset_id)
    with session_scope() as session:
        claimed = queue.claim_next(session, "worker-1")
    with session_scope() as session:
        assert queue.succeed(session, claimed.id, "worker-1", {"answer": "first"}) is True
    with session_scope() as session:
        assert queue.fail(session, claimed.id, "worker-1", "second thoughts") is False

    analysis = load(claimed.id)
    assert analysis.status == AnalysisStatus.SUCCEEDED
    assert analysis.error is None


def test_fail_records_the_reason(dataset_id):
    add(dataset_id)
    with session_scope() as session:
        claimed = queue.claim_next(session, "worker-1")
    with session_scope() as session:
        queue.fail(session, claimed.id, "worker-1", "the column does not exist")

    analysis = load(claimed.id)
    assert analysis.status == AnalysisStatus.FAILED
    assert "column does not exist" in analysis.error
    assert analysis.result is None


# ---------------------------------------------------------------- cancellation


def test_cancelling_a_queued_analysis_stops_it_immediately(dataset_id):
    analysis_id = add(dataset_id)
    with session_scope() as session:
        assert queue.cancel(session, analysis_id) == AnalysisStatus.CANCELLED

    assert load(analysis_id).status == AnalysisStatus.CANCELLED
    with session_scope() as session:
        assert queue.claim_next(session, "worker-1") is None


def test_cancelling_a_running_analysis_only_sets_a_flag(dataset_id):
    """The worker owns the work, so it is the only thing that can stop it cleanly.

    Flipping the status to CANCELLED here would leave the worker running, and its
    result would then arrive for a job the UI already reported as cancelled.
    """
    add(dataset_id)
    with session_scope() as session:
        claimed = queue.claim_next(session, "worker-1")
    with session_scope() as session:
        assert queue.cancel(session, claimed.id) == AnalysisStatus.RUNNING

    assert load(claimed.id).status == AnalysisStatus.RUNNING

    with session_scope() as session:
        outcome = queue.heartbeat(session, claimed.id, "worker-1")
    assert outcome.still_owned is True
    assert outcome.cancel_requested is True

    with session_scope() as session:
        assert queue.mark_cancelled(session, claimed.id, "worker-1") is True
    assert load(claimed.id).status == AnalysisStatus.CANCELLED


def test_cancelling_a_finished_analysis_changes_nothing(dataset_id):
    add(dataset_id)
    with session_scope() as session:
        claimed = queue.claim_next(session, "worker-1")
    with session_scope() as session:
        queue.succeed(session, claimed.id, "worker-1", {"answer": "done"})
    with session_scope() as session:
        assert queue.cancel(session, claimed.id) == AnalysisStatus.SUCCEEDED
    assert load(claimed.id).status == AnalysisStatus.SUCCEEDED


def test_cancelling_something_that_does_not_exist_returns_none(db):
    with session_scope() as session:
        assert queue.cancel(session, uuid.uuid4()) is None


# ------------------------------------------------------------------- reclaiming


def test_a_live_job_is_not_reclaimed(dataset_id):
    """The timeout has to mean something. A job claimed one second ago with a fresh
    heartbeat must survive a sweep, or a healthy worker loses its work."""
    add(dataset_id)
    with session_scope() as session:
        claimed = queue.claim_next(session, "worker-1")
    with session_scope() as session:
        report = queue.reclaim_stalled(session, timeout_s=999, max_attempts=3)

    assert not report
    assert load(claimed.id).status == AnalysisStatus.RUNNING


def test_a_stalled_job_is_requeued_and_stripped_of_its_worker(dataset_id):
    add(dataset_id)
    with session_scope() as session:
        claimed = queue.claim_next(session, "worker-1")
    with session_scope() as session:
        report = queue.reclaim_stalled(session, timeout_s=0, max_attempts=3)

    assert report.requeued == [claimed.id]
    assert report.abandoned == []

    analysis = load(claimed.id)
    assert analysis.status == AnalysisStatus.PENDING
    assert analysis.worker_id is None
    assert analysis.heartbeat_at is None
    assert analysis.started_at is None
    # The attempt is NOT given back. That is what makes the retry limit a limit.
    assert analysis.attempts == 1
    assert EventKind.RECLAIMED in kinds(claimed.id)


def test_a_job_that_keeps_killing_workers_is_eventually_failed(dataset_id):
    """Without this, a job that reliably crashes its worker is retried until the heat
    death of the universe, taking a worker with it every time."""
    analysis_id = add(dataset_id)

    for _ in range(3):
        with session_scope() as session:
            queue.claim_next(session, "doomed")
        with session_scope() as session:
            report = queue.reclaim_stalled(session, timeout_s=0, max_attempts=3)

    assert report.abandoned == [analysis_id]
    analysis = load(analysis_id)
    assert analysis.status == AnalysisStatus.FAILED
    assert analysis.attempts == 3
    assert "abandoned after 3 attempts" in analysis.error

    with session_scope() as session:
        assert queue.claim_next(session, "another") is None


def test_a_pending_job_is_never_reclaimed(dataset_id):
    """Only RUNNING rows can be stalled. A PENDING row with no heartbeat is not
    orphaned, it is waiting — and requeueing it would reset nothing and log noise."""
    analysis_id = add(dataset_id)
    with session_scope() as session:
        assert not queue.reclaim_stalled(session, timeout_s=0, max_attempts=3)
    assert load(analysis_id).status == AnalysisStatus.PENDING


def test_sweeping_twice_reclaims_once(dataset_id):
    """Every worker sweeps, so the sweep has to be idempotent."""
    add(dataset_id)
    with session_scope() as session:
        queue.claim_next(session, "worker-1")
    with session_scope() as session:
        first = queue.reclaim_stalled(session, timeout_s=0, max_attempts=3)
    with session_scope() as session:
        second = queue.reclaim_stalled(session, timeout_s=0, max_attempts=3)

    assert len(first.requeued) == 1
    assert not second
