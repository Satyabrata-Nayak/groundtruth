"""The job queue: the only thing the API and the worker share.

Placed in its own package rather than in `app/worker/` on purpose. Both processes
need it and neither owns it — the API writes rows and reads status, the worker
claims rows and writes results — so putting it inside either one would make that
process import-time infrastructure for the other.
"""

from app.jobs.queue import (
    ClaimedAnalysis,
    HeartbeatOutcome,
    ReclaimReport,
    cancel,
    claim_next,
    emit,
    enqueue,
    fail,
    heartbeat,
    mark_cancelled,
    reclaim_stalled,
    succeed,
)

__all__ = [
    "ClaimedAnalysis",
    "HeartbeatOutcome",
    "ReclaimReport",
    "cancel",
    "claim_next",
    "emit",
    "enqueue",
    "fail",
    "heartbeat",
    "mark_cancelled",
    "reclaim_stalled",
    "succeed",
]
