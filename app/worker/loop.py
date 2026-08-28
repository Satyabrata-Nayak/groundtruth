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

from app.agent import memory, rewrite
from app.agent.contract import Emit
from app.config import get_settings
from app.db.models import EventKind
from app.db.session import session_scope
from app.jobs import cache, notify, queue
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
        """Claim and run jobs until asked to stop.

        Idle time is spent blocked on a LISTEN connection rather than sleeping, so a
        question asked now is claimed now instead of up to `poll_interval_s` later. The
        interval survives as the timeout on that wait, which is what makes the
        notification an optimisation rather than a dependency: if it is lost, dropped,
        or sent while this worker was busy, the wait times out and the poll finds the
        row anyway.
        """
        log.info("worker %s started", self.worker_id)
        with notify.WorkSignal(fallback_interval_s=self.poll_interval_s) as signal_:
            while not self._shutdown.is_set():
                try:
                    if not self.tick():
                        # Nothing to do. Block on the socket, waking on a notification
                        # or on the fallback timeout — but check the shutdown flag
                        # first, so a signal arriving mid-idle is not held up by it.
                        if self._shutdown.wait(0):
                            break
                        signal_.wait()
                except Exception:  # noqa: BLE001
                    # A crash in the loop itself — Postgres restarted, say — must not
                    # kill the worker. Back off one interval and try again; the job it
                    # was holding, if any, is recovered by the sweep.
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

        # --- before any model runs: has this exact question already been answered? ---
        #
        # The result is a pure function of (dataset, version, question, model), and all
        # four are known here. A hit turns 90-190 seconds of GPU into a primary-key
        # lookup. Done outside the Heartbeat block because there is nothing to keep
        # alive: the whole path is a few milliseconds.
        question, history = self._prepare(claimed, emit)
        if (cached := self._cached(claimed, question, emit)) is not None:
            self._succeed(claimed, cached)
            return

        with Heartbeat(claimed.id, self.worker_id, interval_s=self.heartbeat_interval_s) as beat:
            try:
                result = run_analysis(
                    dataset_id=claimed.dataset_id,
                    version=claimed.dataset_version,
                    question=question,
                    emit=emit,
                    checkpoint=beat.checkpoint,
                    # What the asker chose, pinned on the row at ask time. None for
                    # either falls back to this worker's configuration.
                    llm_model=claimed.llm_model,
                    llm_thinking=claimed.llm_thinking,
                    history=history,
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

            self._remember(claimed, question, result)
            self._succeed(claimed, result)

    # -- memory, rewriting and the cache -----------------------------------

    def _prepare(self, claimed: queue.ClaimedAnalysis, emit: Emit) -> tuple[str, str]:
        """Resolve the question against its thread, and render the history block.

        Two things happen here and they are separate on purpose. The REWRITE makes the
        question stand alone, which is what lets the cache key mean anything — "what
        about France?" hashes differently in every thread, and its standalone form
        hashes the same as somebody asking it directly. The HISTORY still goes to the
        analyst afterwards, because a resolved question can still benefit from knowing
        what was already established.
        """
        if claimed.conversation_id is None:
            return claimed.question, ""

        with session_scope() as session:
            turns = memory.recent_turns(session, claimed.conversation_id)
        if not turns:
            return claimed.question, ""

        question = claimed.question
        if rewrite.needs_rewriting(question, turns):
            resolved = rewrite.rewrite(question, turns, main_model=claimed.llm_model)
            if resolved != question:
                emit(
                    EventKind.NOTE,
                    f"read as: {resolved}",
                    {"original": question, "rewritten": resolved},
                )
                question = resolved

        return question, memory.render(turns)

    def _cached(
        self, claimed: queue.ClaimedAnalysis, question: str, emit: Emit
    ) -> dict[str, Any] | None:
        model = claimed.llm_model or get_settings().llm_model
        with session_scope() as session:
            hit = cache.lookup(
                session,
                dataset_id=claimed.dataset_id,
                dataset_version=claimed.dataset_version,
                question=question,
                llm_model=model,
            )
        if hit is not None:
            emit(
                EventKind.NOTE,
                "answered from cache — this exact question was already computed",
                {"cached": True, "model": model},
            )
        return hit

    def _remember(
        self, claimed: queue.ClaimedAnalysis, question: str, result: dict[str, Any]
    ) -> None:
        with session_scope() as session:
            cache.store(
                session,
                dataset_id=claimed.dataset_id,
                dataset_version=claimed.dataset_version,
                question=question,
                llm_model=claimed.llm_model or get_settings().llm_model,
                result=result,
            )

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
