"""What every analysis engine must satisfy, shared by both of them.

There are two engines now — the fixed one in `app/worker/analysis.py` and the agent in
`app/agent/analyst.py` — and there will be more. They must be interchangeable to the
worker, which means the failure type, the callbacks and the result shape have to live
somewhere neither of them owns. This is that somewhere.

THE RESULT SHAPE
----------------
    {
      "engine":   "agent-v1" | "hardcoded-v1",   what produced this, stored forever
      "question": the question as asked,
      "dataset":  {"id": ..., "version": ...},   what was actually analysed
      "answer":   prose a person reads,
      "steps":    [{tool, arguments, ok, summary, error, duration_ms}, ...],
      "table":    {"columns": [...], "rows": [[...]]} | None,   the evidence
      "chart":    {"chart": {...spec}} | None,
    }

`engine` is stored with every result so a row written by the fixed engine stays
interpretable next to an agent row years later. `steps` and `table` exist because the
answer is a claim and this system's entire premise is that a claim is worth what its
evidence is worth — the UI shows the numbers underneath the sentence so a person can
check one against the other without trusting either.

THE TWO CALLBACKS
-----------------
`emit` records something that happened. `checkpoint` raises if the work should stop.
Both are injected rather than imported so an engine has no opinion about transactions,
threads or the job queue — which is what makes an engine testable with two lambdas and
no worker, no Postgres and no model.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from app.db.models import EventKind

Emit = Callable[[EventKind, str, dict[str, Any] | None], None]
Checkpoint = Callable[[], None]


class AnalysisFailed(Exception):
    """The analysis cannot produce an answer, for a reason worth showing the user.

    Distinct from an unexpected exception, which is a bug in us. This one is written
    to be read by whoever asked the question: "the language model is not running" and
    "this dataset has no numeric columns" are both legitimate outcomes of asking, and
    neither should surface as a traceback.
    """


__all__ = ["AnalysisFailed", "Checkpoint", "Emit", "EventKind"]
