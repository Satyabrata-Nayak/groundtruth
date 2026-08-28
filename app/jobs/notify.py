"""Waking a worker the instant a question is asked, instead of up to a second later.

WHAT THIS IS AND, MORE IMPORTANTLY, WHAT IT IS NOT
---------------------------------------------------
`LISTEN`/`NOTIFY` is a **wake-up signal**. It is not the queue, and building a queue out
of it is the classic mistake: notifications are not persisted, so anything sent while no
worker is listening is gone forever. A worker that restarts during a quiet moment would
never learn about the jobs it missed.

So the queue stays exactly what it was — a table, claimed with `FOR UPDATE SKIP LOCKED`
— and this only removes the *waiting*. The worker blocks on the socket instead of
sleeping a second between polls, and the poll interval survives as a fallback: if a
notification is lost, dropped or arrives while the worker is busy, the next poll finds
the row anyway. Correctness never depends on the notification arriving.

    without   ask → row committed → ...up to 1s of sleep... → poll → claim
    with      ask → row committed → NOTIFY → worker wakes → claim

WHY A DEDICATED CONNECTION
--------------------------
`LISTEN` occupies a connection for as long as it is listening. Taking one from the
application's pool would remove it from circulation and, worse, hand it back to the pool
still subscribed. This opens its own connection, outside SQLAlchemy, and closes it on
shutdown.

WHY THE PAYLOAD IS EMPTY
------------------------
It is tempting to send the analysis id and skip the claim. That would be a second,
unreliable queue: the notification can be lost, two workers can both receive it, and the
payload caps at 8,000 bytes. The signal says "something changed, go and look" and
nothing more — the database remains the only source of truth about what to work on.
"""

from __future__ import annotations

import logging
import select
from typing import Any

import psycopg
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.config import get_settings

log = logging.getLogger(__name__)

# One channel. Postgres channel names are identifiers, so this must never be built from
# user input.
CHANNEL = "analyses_new"


def notify_new_work(session: Session) -> None:
    """Announce that a job was enqueued. Fires when the transaction commits.

    Deliberately not `pg_notify(channel, payload)` with an id — see the module note.
    A best-effort signal: if this fails, the worker's poll still finds the row, so the
    failure is logged and swallowed rather than turning a successful enqueue into a 500.
    """
    try:
        session.execute(text(f"NOTIFY {CHANNEL}"))
    except Exception:  # noqa: BLE001 - a lost wake-up must never fail an enqueue
        log.debug("could not send %s notification", CHANNEL, exc_info=True)


class WorkSignal:
    """A dedicated listening connection, with a poll interval as its safety net.

    Used as a context manager by the worker loop. If the connection cannot be opened —
    an older Postgres, a proxy that does not pass notifications, no permission — every
    method degrades to plain sleeping, so the worker keeps working and only loses the
    latency improvement.
    """

    def __init__(self, *, fallback_interval_s: float) -> None:
        self.fallback_interval_s = fallback_interval_s
        self._connection: psycopg.Connection[Any] | None = None

    def __enter__(self) -> WorkSignal:
        settings = get_settings()
        try:
            self._connection = psycopg.connect(
                host=settings.postgres_host,
                port=settings.postgres_port,
                dbname=settings.postgres_db,
                user=settings.postgres_user,
                password=settings.postgres_password,
                connect_timeout=settings.db_connect_timeout_s,
                # LISTEN has to be visible immediately, and an open transaction would
                # hold it until commit.
                autocommit=True,
            )
            self._connection.execute(f"LISTEN {CHANNEL}")
            log.info(
                "listening on %s; polling every %.1fs as a fallback",
                CHANNEL,
                self.fallback_interval_s,
            )
        except Exception:  # noqa: BLE001 - degrade to polling, never fail to start
            log.warning("could not LISTEN; falling back to polling only", exc_info=True)
            self._connection = None
        return self

    def __exit__(self, *exc_info: object) -> None:
        if self._connection is not None:
            try:
                self._connection.close()
            except Exception:  # noqa: BLE001
                pass
            self._connection = None

    def wait(self, timeout_s: float | None = None) -> bool:
        """Block until work is announced or the timeout expires. True if woken.

        `select` on the connection's file descriptor rather than a library helper: it is
        the same call underneath, it returns the moment the socket is readable, and it
        cannot block past the timeout — which is what makes shutdown responsive.
        """
        timeout = self.fallback_interval_s if timeout_s is None else timeout_s
        if self._connection is None:
            # No listener. The caller still needs to not spin.
            import time

            time.sleep(timeout)
            return False

        try:
            readable, _, _ = select.select([self._connection], [], [], timeout)
            if not readable:
                return False
            # Drain everything queued: several questions asked together should cost one
            # wake-up, not one loop iteration each.
            self._connection.execute("SELECT 1")
            notifications = list(self._connection.notifies(timeout=0))
            return bool(notifications) or True
        except Exception:  # noqa: BLE001 - a broken listener must not stop the worker
            log.warning("listen connection failed; polling from here", exc_info=True)
            self.__exit__()
            return False
