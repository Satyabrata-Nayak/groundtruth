"""Proof of life for a running analysis, sent from a background thread.

WHY A THREAD AND NOT A BEAT BETWEEN STEPS
-----------------------------------------
The obvious design is to call `heartbeat()` between analysis steps. It is simpler and
it is wrong for the case that matters: a single step can be the slow one. In M5 one
model call takes 10-60 seconds; a DuckDB scan over a large Parquet file can take
longer. Beating only between steps means the gap between beats is the duration of the
longest step — precisely when the worker most needs to be saying "still alive".

So the beat runs on its own thread at a fixed interval, independent of what the work
is doing. The worker's job is to *check* the thread's findings at safe points.

WHY THE THREAD HAS ITS OWN SESSION
----------------------------------
A SQLAlchemy `Session` is not thread-safe. Two threads sharing one would interleave
statements on a single DBAPI connection and corrupt the transaction state in ways
that surface as unrelated errors much later. The thread opens its own short
transaction per beat, from the same process-wide pool.

WHY IT LISTENS AS WELL AS SPEAKS
--------------------------------
The same round trip that says "alive" reads back two facts (see `queue.heartbeat`):
whether this worker still owns the row, and whether a cancellation was requested.
Both mean *stop*, for different reasons, and both are recorded as flags the worker
polls rather than exceptions raised across a thread boundary — an exception raised in
this thread could not interrupt the work anyway, and pretending otherwise would hide
the fact that the work only stops at checkpoints.
"""

from __future__ import annotations

import logging
import threading
import uuid

from app.db.session import get_sessionmaker
from app.jobs import queue

log = logging.getLogger(__name__)


class StopRequested(Exception):
    """Raised at a checkpoint when the analysis should not continue.

    `reason` is either "cancelled" (someone asked) or "lost" (this worker no longer
    owns the row, because the sweep reclaimed it). The distinction decides whether
    anything gets written at the end: a cancelled job is reported as CANCELLED, a
    lost one is reported as nothing at all, because another worker owns that story.
    """

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class Heartbeat:
    """A context manager that beats for one analysis while the body runs.

        with Heartbeat(analysis_id, worker_id, interval_s=5) as beat:
            ...work...
            beat.checkpoint()      # raises StopRequested if it should stop
            ...more work...

    The thread is a daemon so a worker that is shutting down is never held open by
    it, and `__exit__` always joins with a timeout so an unresponsive beat cannot
    wedge the loop.
    """

    def __init__(self, analysis_id: uuid.UUID, worker_id: str, *, interval_s: float) -> None:
        self.analysis_id = analysis_id
        self.worker_id = worker_id
        self.interval_s = interval_s
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.cancelled = False
        self.lost_ownership = False

    def __enter__(self) -> Heartbeat:
        self._thread = threading.Thread(
            target=self._run, name=f"heartbeat-{self.analysis_id}", daemon=True
        )
        self._thread.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self._stop.set()
        if self._thread is not None:
            # Twice the interval: long enough for a beat in flight to finish, short
            # enough that a hung connection does not stall the whole worker loop.
            self._thread.join(timeout=self.interval_s * 2)

    def _run(self) -> None:
        session_factory = get_sessionmaker()
        # `wait` returns True the moment `_stop` is set, so shutdown is immediate
        # instead of waiting out a sleep. Beating first, then waiting, means a very
        # short analysis still refreshes the timestamp at least once.
        while True:
            try:
                session = session_factory()
                try:
                    outcome = queue.heartbeat(session, self.analysis_id, self.worker_id)
                    session.commit()
                finally:
                    session.close()

                if not outcome.still_owned:
                    self.lost_ownership = True
                    return
                if outcome.cancel_requested:
                    self.cancelled = True
                    return
            except Exception:  # noqa: BLE001
                # A failed beat is not fatal: the database may be briefly unreachable,
                # and the reclaim threshold is several intervals wide precisely so a
                # transient failure does not cost the job. Log and beat again.
                log.warning("heartbeat failed for %s", self.analysis_id, exc_info=True)

            if self._stop.wait(self.interval_s):
                return

    def checkpoint(self) -> None:
        """Stop here if the analysis should not continue. Cheap: reads two booleans.

        Call it between steps. It deliberately does not interrupt a step already in
        flight — a half-executed SQL statement has no safe place to be abandoned, and
        the caps in `app.data.sandbox` already bound how long one can run.
        """
        if self.cancelled:
            raise StopRequested("cancelled")
        if self.lost_ownership:
            raise StopRequested("lost")
