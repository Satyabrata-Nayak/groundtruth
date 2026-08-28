"""The crash drill: hard-kill a worker mid-analysis and prove the job is not lost.

This is M4 exit criterion 5, and it is the one requirement that cannot be met by
reading the code. Everything about recovery is a claim about what happens when a
process stops existing between two statements, and the only way to check that claim is
to make a process stop existing.

    1. a real worker subprocess claims a real job
    2. it is killed with no chance to clean up          <- TerminateProcess / SIGKILL
    3. the row is left RUNNING, owned by a worker that no longer exists
    4. a sweep with a realistic timeout leaves it alone  <- the heartbeat is recent
    5. once the heartbeat is stale, the sweep requeues it
    6. a healthy worker picks it up and finishes it

WHY THE SUBPROCESS SUBSTITUTES THE ANALYSIS BODY
------------------------------------------------
The real analysis takes about 200 ms, so "kill it in the middle" would be a race with
itself — passing or failing depending on how busy the machine is, which is worse than
no test. The subprocess therefore replaces `run_analysis` with one that sleeps, and
leaves EVERYTHING else real: the claim, the heartbeat thread, the ownership guard, the
sweep and the second worker's completion.

What is under test is the queue's behaviour around a dead process, not what the dead
process was computing. Substituting the part that is irrelevant is what makes the rest
deterministic.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
import uuid

import pytest

from app.data.service import create_dataset
from app.db.models import Analysis, AnalysisStatus, EventKind
from app.db.session import session_scope
from app.jobs import queue
from app.worker.loop import Worker

pytestmark = pytest.mark.integration

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Runs in the subprocess. Everything here is the real thing except the sleeping
# stand-in for the analysis body.
CRASHING_WORKER = """
import time
from app.worker import loop

def never_finishes(**kwargs):
    kwargs["emit"](loop.EventKind.NOTE, "started work that will never finish", None)
    time.sleep(120)

loop.run_analysis = never_finishes
loop.Worker(worker_id="crash-drill-worker").tick()
"""


@pytest.fixture
def dataset(db, data_root, tmp_path):
    path = tmp_path / "sales.csv"
    path.write_text(
        "region,revenue\nWest,100.0\nEast,80.0\nNorth,60.0\nWest,40.0\n"
        "East,30.0\nNorth,20.0\nWest,10.0\nEast,5.0\n",
        encoding="utf-8",
    )
    return create_dataset(path, name="crash-drill")


def status_of(analysis_id: uuid.UUID) -> str:
    with session_scope() as session:
        return session.get(Analysis, analysis_id).status


def wait_until_running(analysis_id: uuid.UUID, timeout_s: float = 30.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if status_of(analysis_id) == AnalysisStatus.RUNNING:
            return
        time.sleep(0.05)
    raise AssertionError("the worker subprocess never claimed the job")


def test_a_hard_killed_worker_loses_its_job_to_another_one(dataset, data_root):
    with session_scope() as session:
        analysis, _ = queue.enqueue(
            session,
            dataset_id=dataset.dataset_id,
            dataset_version=1,
            question="which region sells most?",
        )
        analysis_id = analysis.id

    # The subprocess needs the same storage directory this test is using, and
    # `data_root` only monkeypatched it inside this process.
    env = {**os.environ, "DATA_DIR": str(data_root)}
    process = subprocess.Popen([sys.executable, "-c", CRASHING_WORKER], cwd=REPO_ROOT, env=env)

    try:
        wait_until_running(analysis_id)

        with session_scope() as session:
            claimed = session.get(Analysis, analysis_id)
            assert claimed.attempts == 1
            assert claimed.worker_id == "crash-drill-worker"
            assert claimed.heartbeat_at is not None

        # No SIGTERM, no atexit, no `finally`. The process simply stops, exactly as it
        # would on a power cut or an OOM kill. Anything the worker "would have done on
        # shutdown" does not happen, which is the entire point.
        process.kill()
        process.wait(timeout=30)
    finally:
        if process.poll() is None:
            process.kill()

    # The job is stranded, not lost: still RUNNING, owned by a process that is gone.
    with session_scope() as session:
        stranded = session.get(Analysis, analysis_id)
        assert stranded.status == AnalysisStatus.RUNNING
        assert stranded.result is None
        assert stranded.error is None

    # A sweep with a realistic timeout must NOT touch it — the last heartbeat was a
    # moment ago. If this were reclaimed, the timeout would be worthless and healthy
    # workers would have their jobs stolen out from under them.
    with session_scope() as session:
        assert not queue.reclaim_stalled(session, timeout_s=300, max_attempts=3)
    assert status_of(analysis_id) == AnalysisStatus.RUNNING

    # Once the heartbeat is old enough, it is requeued. `timeout_s=0` stands in for
    # waiting out the real 30 seconds; the comparison it exercises is identical.
    with session_scope() as session:
        report = queue.reclaim_stalled(session, timeout_s=0, max_attempts=3)
    assert report.requeued == [analysis_id]

    with session_scope() as session:
        requeued = session.get(Analysis, analysis_id)
        assert requeued.status == AnalysisStatus.PENDING
        assert requeued.worker_id is None
        assert requeued.attempts == 1  # the dead attempt still counts

    # A healthy worker finishes what the dead one started.
    assert Worker(worker_id="rescuer").tick() is True

    with session_scope() as session:
        finished = session.get(Analysis, analysis_id)
        assert finished.status == AnalysisStatus.SUCCEEDED
        assert finished.attempts == 2
        assert finished.result["engine"] == "hardcoded-v1"
        assert "West" in finished.result["answer"]

        kinds = [event.kind for event in finished.events]

    # The trail tells the whole story, including the part nobody was watching.
    assert kinds.count(EventKind.CLAIMED) == 2
    assert EventKind.RECLAIMED in kinds
    assert kinds[-1] == EventKind.SUCCEEDED


def test_the_killed_worker_cannot_come_back_and_overwrite_the_result(dataset):
    """The other half of the guarantee, without needing a resurrected process.

    A worker does not have to be dead to lose its job — it only has to be slow enough
    to miss its heartbeats. When it wakes up and writes, that write must land on
    nothing.
    """
    with session_scope() as session:
        analysis, _ = queue.enqueue(
            session, dataset_id=dataset.dataset_id, dataset_version=1, question="q"
        )
        analysis_id = analysis.id

    with session_scope() as session:
        queue.claim_next(session, "slow-worker")
    with session_scope() as session:
        queue.reclaim_stalled(session, timeout_s=0, max_attempts=3)
    with session_scope() as session:
        queue.claim_next(session, "fast-worker")

    with session_scope() as session:
        assert queue.succeed(session, analysis_id, "fast-worker", {"answer": "correct"}) is True
    with session_scope() as session:
        assert queue.succeed(session, analysis_id, "slow-worker", {"answer": "stale"}) is False

    with session_scope() as session:
        assert session.get(Analysis, analysis_id).result == {"answer": "correct"}
