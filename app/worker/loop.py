"""The worker loop: claim a job, run it, record what happened, repeat.

    ┌─ sweep for jobs whose worker died ──────────────────┐
    │                                                     │
    ├─ claim_next() ──► nothing?  sleep(poll_interval) ───┤
    │       │                                             │
    │       ▼ got one                                     │
    │  start heartbeat thread                             │
    │       │                                             │
    │       ▼                                             │
    │  run_analysis(...)  ── checkpoint() between steps   │
    │       │                                             │
    │       ├─ ok            ──► succeed()                │
    │       ├─ AnalysisFailed──► fail()                   │
    │       ├─ cancelled     ──► mark_cancelled()         │
    │       └─ lost ownership──► write nothing            │
    │                                                     │
    └─────────────────────────────────────────────────────┘

FOUR OUTCOMES, NOT TWO
----------------------
"Succeeded or failed" is not enough. A worker also has to handle *cancelled* (someone
asked it to stop) and *lost* (the sweep gave the job to somebody else while this
worker was slow). The lost case is the one that is always forgotten, and the correct
behaviour is counter-intuitive: write nothing at all. Another worker now owns that
analysis, and this one's result — computed correctly, from good data — is unwanted.
Every terminal write in `app.jobs.queue` is guarded so that even a mistake here
cannot land it.

RUNNING WORK OUTSIDE THE CLAIMING TRANSACTION
---------------------------------------------
The claim commits immediately; the analysis then runs with no transaction open. Doing
the work inside the claiming transaction would hold a row lock and a pooled connection
for the entire analysis — a minute in M5 — and no event would be visible to anyone
until the very end, because uncommitted rows are invisible to other sessions. The
liveness signal that replaces the lock is the heartbeat.

WHY EACH EVENT IS ITS OWN TRANSACTION
-------------------------------------
Same reason. The UI polls `/analyses/{id}/events` while the analysis runs, and an
event inside a long transaction does not exist yet as far as that query is concerned.
Short transactions are what make the trail live rather than retrospective.
"""

from __future__ import annotations

import logging
import os
import signal
import socket
import threading
import time
import uuid
from typing import Any

from app.config import get_settings
from app.db.models import EventKind
from app.db.session import session_scope
from app.jobs import queue
from app.worker.analysis import AnalysisFailed, run_analysis
from app.worker.heartbeat import Heartbeat, StopRequested

log = logging.getLogger("app.worker")


def make_worker_id() -> str:
    """Host, pid, and a random suffix.

    The suffix matters more than it looks: a restarted worker can be handed the same
    pid by the OS, and a stale row still naming that pid would then pass an ownership
    guard it should fail. A fresh random suffix per process makes identity unforgeable
    by coincidence.
    """
    return f"{socket.gethostname()}:{os.getpid()}:{uuid.uuid4().hex[:8]}"


class Worker:
    def __init__(self, worker_id: str | None = None) -> None:
        settings = get_settings()
        self.worker_id = worker_id or make_worker_id()
        self.poll_interval_s = settings.worker_poll_interval_s
        self.heartbeat_interval_s = settings.worker_heartbeat_interval_s
        self.heartbeat_timeout_s = settings.worker_heartbeat_timeout_s
        self.max_attempts = settings.analysis_max_attempts
        self._shutdown = threading.Event()
        self._last_sweep = 0.0

    # -- lifecycle ---------------------------------------------------------

    def install_signal_handlers(self) -> None:
        """Ctrl+C and `docker stop` ask for a *graceful* stop.

        The flag stops the loop from claiming anything new; a job already running is
        allowed to finish. Aborting it would be pointless — the row would be reclaimed
        and redone from scratch by the next worker anyway, so tearing it down halfway
        just wastes the work already done.
        """

        def request_stop(signum: int, _frame: Any) -> None:
            log.info("signal %s received; finishing current job then stopping", signum)
            self._shutdown.set()

        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, request_stop)

    def run_forever(self) -> None:
        log.info("worker %s started", self.worker_id)
        while not self._shutdown.is_set():
            try:
                if not self.tick():
                    # Nothing to do. `wait` rather than `sleep` so a signal arriving
                    # mid-idle stops the worker immediately instead of a second later.
                    self._shutdown.wait(self.poll_interval_s)
            except Exception:  # noqa: BLE001
                # A crash in the loop itself — Postgres restarted, say — must not kill
                # the worker. Back off one interval and try again; the job it was
                # holding, if any, is recovered by the sweep.
                log.exception("worker loop error; backing off")
                self._shutdown.wait(self.poll_interval_s)
        log.info("worker %s stopped", self.worker_id)

    def tick(self) -> bool:
        """One iteration. Returns True if a job was processed."""
        self.maybe_sweep()

        with session_scope() as session:
            claimed = queue.claim_next(session, self.worker_id)

        if claimed is None:
            return False

        log.info("claimed %s (attempt %s)", claimed.id, claimed.attempts)
        self.process(claimed)
        return True

    def maybe_sweep(self) -> None:
        """Look for orphaned jobs, at most once per half heartbeat timeout.

        Every worker sweeps; none is elected. A leader would be one more thing to
        elect, monitor and fail over, and the sweep is idempotent — concurrent sweeps
        serialise on the row lock and the loser updates nothing.
        """
        now = time.monotonic()
        if now - self._last_sweep < self.heartbeat_timeout_s / 2:
            return
        self._last_sweep = now

        with session_scope() as session:
            report = queue.reclaim_stalled(
                session,
                timeout_s=self.heartbeat_timeout_s,
                max_attempts=self.max_attempts,
            )
        if report:
            log.warning(
                "reclaimed %d stalled job(s), abandoned %d",
                len(report.requeued),
                len(report.abandoned),
            )

    # -- running one job ---------------------------------------------------

    def process(self, claimed: queue.ClaimedAnalysis) -> None:
        def emit(kind: EventKind, message: str, payload: dict[str, Any] | None = None) -> None:
            with session_scope() as session:
                queue.emit(session, claimed.id, kind, message, payload)

        with Heartbeat(claimed.id, self.worker_id, interval_s=self.heartbeat_interval_s) as beat:
            try:
                result = run_analysis(
                    dataset_id=claimed.dataset_id,
                    version=claimed.dataset_version,
                    question=claimed.question,
                    emit=emit,
                    checkpoint=beat.checkpoint,
                    # What the asker chose, pinned on the row at ask time. None for
                    # either falls back to this worker's configuration.
                    llm_model=claimed.llm_model,
                    llm_thinking=claimed.llm_thinking,
                )
            except StopRequested as stop:
                self._handle_stop(claimed, stop)
                return
            except AnalysisFailed as failure:
                # Bound to a plain string first, deliberately. Python deletes the name
                # `failure` when the except block ends, so a closure that captured it
                # would be a NameError waiting for the one code path that defers the
                # call. Ruff catches this (F821); the interpreter would not, until it
                # mattered.
                message = str(failure)
                log.warning("analysis %s failed: %s", claimed.id, message)
                self._fail(claimed, message)
                return
            except Exception as error:  # noqa: BLE001
                # A bug in us, not in the request. Record it as the failure it is, with
                # the type name, rather than letting the job sit RUNNING until the
                # sweep has to guess what happened to it.
                message = f"internal error: {type(error).__name__}: {error}"
                log.exception("analysis %s crashed", claimed.id)
                self._fail(claimed, message)
                return

            self._succeed(claimed, result)

    def _handle_stop(self, claimed: queue.ClaimedAnalysis, stop: StopRequested) -> None:
        if stop.reason == "cancelled":
            log.info("analysis %s cancelled", claimed.id)
            self._terminal(claimed.id, queue.mark_cancelled, claimed.id, self.worker_id)
            return
        # Lost. Another worker owns this analysis now; anything written here would be
        # a second, conflicting story about the same row.
        log.warning("analysis %s was reclaimed while this worker held it; dropping", claimed.id)

    def _succeed(self, claimed: queue.ClaimedAnalysis, result: dict[str, Any]) -> None:
        self._terminal(claimed.id, queue.succeed, claimed.id, self.worker_id, result)

    def _fail(self, claimed: queue.ClaimedAnalysis, message: str) -> None:
        self._terminal(claimed.id, queue.fail, claimed.id, self.worker_id, message)

    def _terminal(self, analysis_id: uuid.UUID, action: Any, *args: Any) -> None:
        """Apply a terminal write, and say so when the ownership guard refuses it.

        A refusal is not an error to raise: it means the sweep handed this analysis to
        another worker while this one was still working, and the correct response is to
        drop the result quietly. Logging it is how that stays visible rather than
        mysterious.
        """
        with session_scope() as session:
            if not action(session, *args):
                log.warning(
                    "terminal write for %s was rejected: this worker no longer owns it",
                    analysis_id,
                )
